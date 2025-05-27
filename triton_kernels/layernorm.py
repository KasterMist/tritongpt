import torch
import triton
import triton.language as tl
from typing import Tuple
from .reduction import get_optimal_split, column_base_reduction

# ------------------- forward part -------------------

@triton.jit
def layernorm_fwd_kernel(
    output_ptr,
    input_ptr,
    alpha_ptr,          # learnable scale parameter for Affine Transformation (layernorm's weights)
    beta_ptr,           # leanable shift parameter for Affine Transformation (layernorm's biases)
    d_embd,             # the last dim for input_ptr and output_ptr
    input_batch_stride, # stride for input batch
    input_seq_stride,   # stride for input seq 
    BLOCK_SIZE,         # BLOCK_SIZE should larger(or equal) than d_embd
    eps,                # to avlid divided by 0
):
    batch_pid = tl.program_id(axis=1)
    seq_pid = tl.program_id(axis=0)

    input_start_ptr = input_ptr + batch_pid * input_batch_stride + seq_pid * input_seq_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    input_ptrs = input_start_ptr + offsets
    mask = offsets < d_embd

    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    alpha = tl.load(alpha_ptr + offsets, mask)
    beta = tl.load(beta_ptr + offsets, mask=mask)

    # calculate x_mean
    x_mean = tl.sum(x, axis=0) / d_embd
    centered = tl.where(offsets < d_embd, x - x_mean, 0.0)

    # calculate variance
    x_var = tl.sum(centered * centered, axis=0) / d_embd

    # invert standard deviation
    rstd = 1.0 / tl.sqrt(x_var + eps)

    # normalization
    norm = tl.where(offsets < d_embd, centered * rstd, 0.0)
    affine = alpha * norm + beta

    output_start_ptr = output_ptr + batch_pid * input_batch_stride + seq_pid * input_seq_stride
    output_ptrs = output_start_ptr + offsets

    tl.store(output_ptrs, affine, mask=mask)


def layernorm_fwd(
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        eps=1e-05
) -> torch.Tensor:
    output = torch.empty_like(x)
    batch_num, seq_num, d_embd = x.shape
    BLOCK_SIZE = triton.next_power_of_2(d_embd)

    assert x.is_cuda and output.is_cuda and x.dtype == torch.float16

    layernorm_fwd_kernel[(batch_num, seq_num)](
        output_ptr=output,
        input_ptr=x,
        alpha_ptr=alpha,
        beta_ptr=beta,
        d_embd=d_embd,
        input_batch_stride=x.stride(0),
        input_seq_stride=x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps
    )

    return output


# ---------------- backward part ----------------

@triton.jit
def layernorm_bwd_dx_kernel(
    grad_out_ptr,               # output gradient pointer (dLoss/dy)
    input_ptr,                  # input pointer (x)
    alpha_ptr,                  # learnable scale parameter for Affine Transformation (layernorm's weights)
    grad_x_ptr,                 # gradient for input (dLoss/dx)
    grad_out_stride_batch,      # stride for batch dimension in output gradient pointer
    grad_out_stride_seq,        # stride for sequence dimension in output gradient pointer
    input_stride_batch,         # stride for batch dimension in input pointer
    input_stride_seq,           # stride for sequence dimension in input pointer
    d_embd,                     # the last dim for grad_out_ptr and input_ptr 
    eps,                        # to avoid divided by 0
    BLOCK_SIZE: tl.constexpr,   # BLOCK_SIZE should larger(or equal) than d_embd
):
    """
    layer normalization bakward for dLoss/dx.
    Each program will handle a block of d_embd elements.
    """
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_embd

    grad_out_start_ptr = grad_out_ptr + pid_batch * grad_out_stride_batch + pid_seq * grad_out_stride_seq

    alpha = tl.load(alpha_ptr + offsets, mask=mask)
    grad_out = tl.load(grad_out_start_ptr + offsets, mask=mask)

    input_start_ptr = input_ptr + pid_batch * input_stride_batch + pid_seq * input_stride_seq
    input = tl.load(input_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_mean = tl.sum(input, axis=0) / d_embd
    x_centered = tl.where(mask, input - x_mean, 0.0)
    x_var = tl.sum(x_centered * x_centered, axis=0) / d_embd

    """
    Calculation: dLoss/dx_i = rstd * (d_out_i - d_out_mean - 1/d_embd * x_hat_i * rstd * rstd * SUM_j(d_out_j * x_hat_i))
                x_hai_i(x_centered_i) = x_i - x_mean
                d_out_i = alpha_i * dLoss/dy_i
                d_out_mean = SUM_j(d_out_j) / d_embd
    """
    rstd = 1.0 / tl.sqrt(x_var + eps)
    d_out = alpha * grad_out
    c1 = tl.sum(d_out * x_centered) / d_embd  # SUM_j(d_out_j * x_hat_i) / d_embd
    c2 = rstd * (d_out - (tl.math.pow(rstd, 2.0) * x_centered * c1)) # dLoss/dx without minus d_out_mean
    grad_input = c2 - tl.sum(c2) * (1.0 / d_embd)  # tl.sum(2) * (1.0 / d_embd) is equal to d_out_mean

    grad_x_start_ptr = grad_x_ptr + pid_batch * input_stride_batch + pid_seq * input_stride_seq
    tl.store(grad_x_start_ptr + offsets, grad_input, mask=mask)

def layernorm_bwd_dx(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        alpha: torch.tensor,
        eps=1e-05
) -> torch.Tensor:
    batch_num, seq_num, d_embd = grad_output.shape
    grad_input = torch.empty_like(input)

    grid = (batch_num, seq_num)
    BLOCK_SIZE = triton.next_power_of_2(d_embd)

    layernorm_bwd_dx_kernel[grid](
        grad_out_ptr=grad_output,
        input_ptr=input,
        alpha_ptr=alpha,
        grad_x_ptr=grad_input,
        grad_out_stride_batch=grad_output.stride(0),
        grad_out_stride_seq=grad_output.stride(1),
        input_stride_batch=input.stride(0),
        input_stride_seq=input.stride(1),
        d_embd=d_embd,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    return grad_input


@triton.jit
def layernorm_bwd_da_db_kernel(
    grad_output_ptr,            # output gradient pointer (dLoss/dy)
    grad_alpha_ptr,             # gradient for alpha (dLoss/dalpha)
    grad_beta_ptr,              # gradient for beta (dLoss/dbeta)
    input_ptr,                  # input pointer (x)
    grad_output_stride_batch,   # stride for batch dimension in output gradient pointer
    grad_alpha_stride_batch,    # stride for batch dimension in alpha gradient pointer
    grad_beta_stride_batch,     # stride for batch dimension in beta gradient pointer
    input_stride_batch,         # stride for batch dimension in input pointer
    d_embd,                     # the last dim for grad_output_ptr and input_ptr
    batch_num: tl.constexpr,    # batch number
    eps,                        # to avoid divided by 0
    BLOCK_SIZE: tl.constexpr,   # BLOCK_SIZE should larger(or equal) than d_embd
    GROUP_NUM: tl.constexpr,   # number of groups for parallel reduction
):
    batch_pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_embd
    
    alpha_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator_alpha = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    beta_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator_beta = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    for i in range(batch_pid, batch_num, GROUP_NUM):
        grad_output_start_ptr = grad_output_ptr + i * grad_output_stride_batch
        input_start_ptr = input_ptr + i * input_stride_batch

        x = tl.load(input_start_ptr + offsets, mask=mask).to(tl.float32)
        grad_output = tl.load(grad_output_start_ptr + offsets, mask=mask)

        x_mean = tl.sum(x, axis=0) / d_embd
        x_centered = tl.where(mask, x - x_mean, 0.0)
        x_var = tl.sum(x_centered * x_centered, axis=0) / d_embd
        rstd = 1.0 / tl.sqrt(x_var + eps)

        grad_alpha_accum = (grad_output * rstd * x_centered).to(tl.float16)
        grad_beta_accum = grad_output.to(tl.float16)

        y_alpha = grad_alpha_accum - compensator_alpha
        y_beta = grad_beta_accum - compensator_beta

        tmp_alpha = alpha_sum + y_alpha
        compensator_alpha = (tmp_alpha - alpha_sum) - y_alpha
        alpha_sum = tmp_alpha

        tmp_beta = beta_sum + y_beta
        compensator_beta = (tmp_beta - beta_sum) - y_beta
        beta_sum = tmp_beta
    
    tl.store(grad_alpha_ptr + grad_alpha_stride_batch * batch_pid + offsets, alpha_sum, mask=mask)
    tl.store(grad_beta_ptr + grad_beta_stride_batch * batch_pid + offsets, beta_sum, mask=mask)


def layernorm_bwd_da_db(grad_output: torch.Tensor,
                        input: torch.Tensor,
                        eps=1e-05) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.view(-1, input.shape[-1])
    _, max_fact = get_optimal_split(input.shape[-1])
    GROUP_NUM = max_fact

    grad_alpha = torch.empty(
        (GROUP_NUM, input.shape[-1]),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    grad_beta = torch.empty(
        (GROUP_NUM, input.shape[-1]),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    
    grad_output = grad_output.view(-1, grad_output.shape[-1])
    BLOCK_SIZE = triton.next_power_of_2(input.shape[-1])
    layernorm_bwd_da_db_kernel[(GROUP_NUM,)](
        grad_output_ptr=grad_output,
        grad_alpha_ptr=grad_alpha,
        grad_beta_ptr=grad_beta,
        input_ptr=input,
        grad_output_stride_batch=grad_output.stride(0),
        grad_alpha_stride_batch=grad_alpha.stride(0),
        grad_beta_stride_batch=grad_beta.stride(0),
        input_stride_batch=input.stride(0),
        d_embd=input.shape[-1],
        batch_num=input.shape[0],
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_NUM=GROUP_NUM,
        num_warps=4
    )

    return column_base_reduction(grad_alpha), column_base_reduction(grad_beta)