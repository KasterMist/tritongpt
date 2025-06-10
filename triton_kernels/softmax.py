import torch

import triton
import triton.language as tl

# input size should be equal to output size
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, input_col_stride, row_size, col_size,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) # row start idx
    programs_size = tl.num_programs(0) # total number of programs, which is also the row steps for range
    for row_idx in tl.range(pid, row_size, programs_size):
        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        # input_ptrs = row_start_ptr + col_offsets[:, None] * input_col_stride
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < col_size
        input_vals = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        minus_max = input_vals - tl.max(input_vals, axis=0)
        exp_minus_max = tl.exp(minus_max)
        sum_exp = tl.sum(exp_minus_max, axis=0)

        softmax_outputs = exp_minus_max / sum_exp

        output_row_start_ptr = output_ptr + row_idx * input_row_stride
        output_ptrs = output_row_start_ptr + col_offsets

        tl.store(output_ptrs, softmax_outputs, mask=mask)


def softmax(x):
    assert x.is_contiguous()
    assert x.dim() == 2
    row_size, col_size = x.shape[0], x.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(col_size)

    y = torch.empty_like(x)
    softmax_kernel[(256,1, 1)](y, x, x.stride(0), x.stride(1), row_size, col_size, BLOCK_SIZE)

    return y

@triton.jit
def softmax_fwd_kernel(
    output_ptr,
    input_ptr,
    input_batch_stride,
    input_row_stride,
    num_cols,
    BLOCK_SIZE: tl.constexpr
):
    batch_pid = tl.program_id(axis=0)
    row_pid = tl.program_id(axis=1)

    row_start_ptr = input_ptr + batch_pid * input_batch_stride + row_pid * input_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    input_ptrs = row_start_ptr + offsets
    mask = offsets < num_cols

    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_max = tl.max(row, axis=0)
    minus_max = row - row_max
    exp_minus_max = tl.exp(minus_max)
    sum_exp = tl.sum(exp_minus_max, axis=0) 
    result = exp_minus_max / sum_exp

    output_start_ptr = output_ptr + input_batch_stride * batch_pid + row_pid * input_row_stride
    output_ptrs = output_start_ptr + offsets

    tl.store(output_ptrs, result, mask=mask)


def softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    num_batch, num_rows, num_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(num_cols)
    output = torch.empty_like(x)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8

    softmax_fwd_kernel[(num_batch, num_rows)](
        output_ptr=output,
        input_ptr=x,
        input_batch_stride=x.stride(0),
        input_row_stride=x.stride(1),
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return output


@triton.jit
def softmax_bwd_kernel(
    grad_output_ptr,        # dLoss/dσ(z)
    output_ptr,             # σ(z)
    grad_input_ptr,         # dLoss/dz, the target value
    batch_stride,
    row_stride,
    num_cols,
    BLOCK_SIZE: tl.constexpr
):
    batch_pid = tl.program_id(axis=0)
    row_pid = tl.program_id(axis=1)

    grad_output_row_start_ptr = grad_output_ptr + batch_stride * batch_pid + row_stride * row_pid
    output_row_start_ptr = output_ptr + batch_stride * batch_pid + row_stride * row_pid

    offsets = tl.arange(0, BLOCK_SIZE)

    grad_output_ptrs = grad_output_row_start_ptr + offsets
    output_ptrs = output_row_start_ptr + offsets
    
    mask = offsets < num_cols

    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
    output = tl.load(output_ptrs, mask=mask, other=0.0)

    prod = output * grad_output
    tmp = grad_output - tl.sum(prod, axis=0)
    result = output * tmp

    grad_input_row_start_ptr = grad_input_ptr + batch_stride * batch_pid + row_stride * row_pid
    grad_input_ptrs = grad_input_row_start_ptr + offsets

    tl.store(grad_input_ptrs, result, mask=mask)


def softmax_bwd(grad_output: torch.Tensor, saved_output: torch.Tensor) -> torch.Tensor:
    num_batch, num_rows, num_cols = grad_output.shape
    BLOCK_SIZE = triton.next_power_of_2(num_cols)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8
    grad_input = torch.empty_like(grad_output)
    softmax_bwd_kernel[(num_batch, num_rows)](
        grad_output_ptr=grad_output,
        output_ptr=saved_output,
        grad_input_ptr=grad_input,
        batch_stride=grad_output.stride(0),
        row_stride=grad_output.stride(1),
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return grad_input


if __name__ == '__main__':
    torch.manual_seed(0)
    x_1 = torch.randn(1823, 781, device=torch.device('cuda:0'), dtype=torch.float32)
    x_2 = x_1.view(1, 1823, 781)
    y_triton = softmax(x_1)
    y_triton_2 = softmax_fwd(x_2)
    y_torch_1 = torch.softmax(x_1, axis=1)
    y_torch_2 = torch.softmax(x_2, axis=2)
    assert torch.allclose(y_triton, y_torch_1, atol=1e-4)
    assert torch.allclose(y_triton_2, y_torch_2, atol=1e-4)
    print("Triton and PyTorch softmax outputs match!")


