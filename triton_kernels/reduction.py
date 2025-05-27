import torch
import triton
import triton.language as tl
from typing import Tuple

import math
from functools import lru_cache

@lru_cache
def get_optimal_split(num: int) -> Tuple[int, int]:
    """
    Finds the factorization which distributes work evenly in the reduction.
    For example, get_optimal_split(65) = (5, 13), get_optimal_split(67) = (1, 67)
    """
    start = int(math.sqrt(num))
    cond = num % start
    while cond != 0:
        start -= 1
        cond = num % start
    
    return start, num // start


@triton.jit
def column_base_naive_reduction_kernel(
    input_ptr,
    output_ptr,
    input_stride_0, # input's dim 0's stride
    numel: tl.constexpr, # input's shape 1
    batch_num: tl.constexpr, # input's shape 0
    BLOCK_SIZE: tl.constexpr,
):
    """
    Naive reduction using Kahan summation to minimize error
    reduction on column
    """
    pid = tl.program_id(axis=0)
    assert pid == 0 # only one thread block to deal with reduction. thread num is larger than the x.shape[-1]

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    sum = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator = tl.zeros([BLOCK_SIZE], dtype=tl.float16) # The 'compensation term' in Kahan summation is used to reduce floating-point rounding errors.

    for i in range(batch_num):
        input_start_ptr = input_ptr + i * input_stride_0
        y = tl.load(input_start_ptr + offsets, mask=mask) - compensator
        tmp = sum + y                       # i.e. tmp = 1e6 + 0.1 = 1e6
        compensator = (tmp - sum) - y       # compensator = 1e6 - 1e6 - 0.1 = -0.1
        sum = tmp

    tl.store(output_ptr + offsets, sum, mask=mask)


def column_base_naive_reduction(x) -> torch.Tensor:
    out = torch.empty((1, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(x.shape[-1])
    column_base_naive_reduction_kernel[(1, )](
        input_ptr=x,
        output_ptr=out,
        input_stride_0=x.stride(0),
        numel=x.shape[1],
        batch_num=x.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return out


@triton.jit
def column_base_reduction_kernel(
    input_ptr,
    output_ptr,
    input_stride_0, # input's dim 0's stride
    numel: tl.constexpr, # input's shape 1
    batch_num: tl.constexpr, # input's shape 0
    BLOCK_SIZE: tl.constexpr,
    GROUP_NUM: tl.constexpr,
):
    """
    parallel reduction 1 to get a [GROUP_NUM, numel] intermediate output,  
    """
    pid = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    sum = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    for i in range(pid, batch_num, GROUP_NUM):
        input_start_ptr = input_ptr + i * input_stride_0
        y = tl.load(input_start_ptr + offsets, mask=mask) - compensator
        tmp = sum + y
        compensator = (tmp - sum) - y 
        sum = tmp
    
    tl.store(output_ptr + pid * input_stride_0 + offsets, sum, mask=mask)


def column_base_reduction(x) -> torch.Tensor:
    """
    Get [1, 1, numel] final sum output.
    """
    x = x.view(-1, x.shape[-1])
    _, max_fact = get_optimal_split(x.shape[0])
    GROUP_NUM = max_fact

    tmp = torch.empty((GROUP_NUM, x.shape[-1]), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(x.shape[-1])

    column_base_reduction_kernel[(GROUP_NUM,)](
        input_ptr=x,
        output_ptr=tmp,
        input_stride_0=x.stride(0),
        numel=x.shape[1],
        batch_num=x.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_NUM=GROUP_NUM,
        num_warps=4
    )

    return column_base_naive_reduction(tmp)



if __name__ == '__main__':
    input = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    result = column_base_reduction(input)
    result_compare = input.sum(axis=0, keepdim=True).view(1, 1, 512)
    assert torch.allclose(result, result_compare, atol=1e-1)
    print("simple assert success")