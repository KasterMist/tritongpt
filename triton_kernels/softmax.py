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

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=torch.device('cuda:0'), dtype=torch.float32)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-4)
    print("Triton and PyTorch softmax outputs match!")


