import torch

import triton
import triton.language as tl

"""
Triton kernels for matrix multiplication (stored in float16)
"""

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, # matrix a, b, c's pointers
    M, N, K, # matrix dimensions [M, K] @ [K, N]
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, # strides
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fuse_relu: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_programs_in_group = num_programs_n * GROUP_SIZE_M
    group_id = pid // num_programs_in_group 

    first_pid_m = group_id *  GROUP_SIZE_M # index of first program id in current group
    group_size_m = min(num_programs_m - first_pid_m, GROUP_SIZE_M)

    pid_offset = pid - group_id * num_programs_in_group
    pid_m = pid_offset % group_size_m + first_pid_m
    pid_n = pid_offset // group_size_m

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_masks = (offsets_m[:, None] < M) & (offsets_k[None, :] < K - i * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_masks, other=0.0)

        b_masks = (offsets_k[:, None] < K - i * BLOCK_SIZE_K) & (offsets_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_masks, other=0.0)

        c += tl.dot(a, b, allow_tf32=False)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if fuse_relu: 
        c = tl.maximum(c, 0.0)
    c_ptrs = c_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    c_masks = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_masks)


def matmul(a: torch.Tensor,
       b: torch.Tensor,
       fuse_relu: bool = False) -> torch.Tensor:
    """
    Performs a matrix-multiply between batched tensor a and b.
    - a (M,K) and b (K,N)
    - Returns a tensor of shape (M, N)

    Optionally supports fusing ReLU activation computation.
    """
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META : (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        fuse_relu=fuse_relu,
    )
    return c


@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ],
        key=["dim_m", "dim_k", "dim_n"],
)
@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_batch_stride, a_row_stride, a_col_stride,
    b_batch_stride, b_row_stride, b_col_stride,
    c_batch_stride, c_row_stride, c_col_stride,
    dim_m, dim_k, dim_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, # number of programs(in row) in a group
):
    batch_pid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_col

    group_id = pid // num_pid_in_group
    first_pid_row_in_group = group_id * GROUP_SIZE_M                            # index of the first program id in the current group
    group_size_row = min(num_pid_row - first_pid_row_in_group, GROUP_SIZE_M)    # actual number of programs (in row) in the current group
    pid_row = first_pid_row_in_group + ((pid - num_pid_in_group * group_id) % group_size_row) # program id in row (column-major order --> [0, 0], [1, 0], [2, 0]...)
    pid_col = (pid % num_pid_in_group) // group_size_row                        # program id in column

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + batch_pid * a_batch_stride,
        shape=(dim_m, dim_k),
        strides=(a_row_stride, a_col_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + batch_pid * b_batch_stride,
        shape=(dim_k, dim_n),
        strides=(b_row_stride, b_col_stride),
        offsets=(0, pid_col * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + batch_pid * c_batch_stride,
        shape=(dim_m, dim_n),
        strides=(c_row_stride, c_col_stride),
        offsets=(pid_row * BLOCK_SIZE_M, pid_col * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )

    for k in range(0, tl.cdiv(dim_k, BLOCK_SIZE_K)):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1)) # check boundary for dim0 and dim1
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_block, b_block)

        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_SIZE_K, 0))
    
    acc = acc.to(tl.float16) # convert to float16 for better performance
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs a batch matrix-multiply.
    a (B, M, K) and b (B, K, N)
    returns a tensor of shape (B, M, N)
    """
    assert(a.shape[-1] == b.shape[-2]), f"Dimension mismatch. Expected a.shape[2] ({a.shape[-1]}) to be equal to b.shape[0] ({b.shape[-2]})"
    assert a.ndim == 3 and b.ndim == 3, "Incorrect number of dimensions. Expected 3D tensors."

    B, M, K, N = a.shape[0], a.shape[1], a.shape[2], b.shape[2]
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    assert a.is_cuda and b.is_cuda and c.is_cuda, "All tensors must be on CUDA device."

    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]))

    bmm_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        M, K, N,
    )

    return c


@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ],
        key=["dim_m", "dim_k", "dim_n"],
)
@triton.jit
def mm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_batch_stride, a_row_stride, a_col_stride,
    b_row_stride, b_col_stride,
    c_batch_stride, c_row_stride, c_col_stride,
    dim_m, dim_k, dim_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, # number of programs(in row) in a group
    fuse_relu: tl.constexpr = False, # whether to fuse ReLU activation
):
    batch_pid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_col

    group_id = pid // num_pid_in_group
    first_pid_row_in_group = group_id * GROUP_SIZE_M
    group_size_row = min(GROUP_SIZE_M, num_pid_row - first_pid_row_in_group)
    pid_row = first_pid_row_in_group + ((pid - num_pid_in_group * group_id) % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + batch_pid * a_batch_stride,
        shape=(dim_m, dim_k),
        strides=(a_row_stride, a_col_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(dim_k, dim_n),
        strides=(b_row_stride, b_col_stride),
        offsets=(0, pid_col * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + batch_pid * c_batch_stride,
        shape=(dim_m, dim_n),
        strides=(c_row_stride, c_col_stride),
        offsets=(pid_row * BLOCK_SIZE_M, pid_col * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )

    for k in range(0, tl.cdiv(dim_k, BLOCK_SIZE_K)):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1)) # check boundary for dim0 and dim1
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_block, b_block)

        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_SIZE_K, 0))
    
    if fuse_relu:
        acc = tl.maximum(acc, 0.0)

    acc = acc.to(tl.float16) # convert to float16 for better performance
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def mm(a: torch.Tensor, b: torch.Tensor, fuse_relu: bool = False) -> torch.Tensor:
    """
    Performs a matrix-multiply between batched tensor a and b.
    - a (B, M, K) and b (K, N)
    - Returns a tensor of shape (B, M, N)

    Optionally supports fusing ReLU activation computation.
    """
    assert a.shape[-1] == b.shape[0], f"Dimension mismatch. Expected a.shape[2] ({a.shape[-1]}) to be equal to b.shape[0] ({b.shape[0]})"
    assert a.ndim == 3 and b.ndim == 2, "Incorrect number of dimensions. Expected 3D tensor for a and 2D tensor for b."

    B, M, K = a.shape
    K, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    assert a.is_cuda and b.is_cuda and c.is_cuda, "All tensors must be on CUDA device."

    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]))

    mm_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        M, K, N,
        fuse_relu=fuse_relu,
    )

    return c


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    a_batch_stride, a_row_stride, a_col_stride,
    b_batch_stride, b_row_stride, b_col_stride,
    c_batch_stride, c_row_stride, c_col_stride,
    dim_m, dim_k, dim_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, # number of programs(in row) in a group
    fuse_relu: tl.constexpr = False, # whether to fuse ReLU activation
):
    batch_pid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_col

    group_id = pid // num_pid_in_group
    first_pid_row_in_group = group_id * GROUP_SIZE_M
    group_size_row = min(GROUP_SIZE_M, num_pid_row - first_pid_row_in_group)
    pid_row = first_pid_row_in_group + ((pid - num_pid_in_group * group_id) % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + batch_pid * a_batch_stride,
        shape=(dim_m, dim_k),
        strides=(a_row_stride, a_col_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + batch_pid * b_batch_stride,
        shape=(dim_k, dim_n),
        strides=(b_row_stride, b_col_stride),
        offsets=(0, pid_col * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + batch_pid * c_batch_stride,
        shape=(dim_m, dim_n),
        strides=(c_row_stride, c_col_stride),
        offsets=(pid_row * BLOCK_SIZE_M, pid_col * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )

    bias_start = pid_col * BLOCK_SIZE_K
    offsets = bias_start + tl.arange(0, BLOCK_SIZE_K)

    for k in range(0, tl.cdiv(dim_k, BLOCK_SIZE_K)):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1)) # check boundary for dim0 and dim1
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_block, b_block)

        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_SIZE_K, 0))
    
    bias = tl.load(bias_ptr + offsets, mask=offsets < dim_k)
    acc = acc + bias

    if fuse_relu:
        acc = tl.where(acc > 0, acc, 0.0)

    acc = acc.to(tl.float16) # convert to float16 for better performance
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, fuse_relu: bool = False) -> torch.Tensor:
    """
    Performs a matrix-multiply between batched tensor a and b with an optional bias.
    - a (B, M, K), b (B, K, N), bias (N)
    - Returns a tensor of shape (B, M, N)

    Optionally supports fusing ReLU activation computation.
    """
    assert a.shape[-1] == b.shape[-2], f"Dimension mismatch. Expected a.shape[2] ({a.shape[-1]}) to be equal to b.shape[0] ({b.shape[-2]})"
    assert a.ndim == 3 and b.ndim == 3 and bias.ndim == 1, "Incorrect number of dimensions. Expected 3D tensors for a and b, and 1D tensor for bias."

    B, M, K = a.shape
    K, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    assert a.is_cuda and b.is_cuda and c.is_cuda and bias.is_cuda, "All tensors must be on CUDA device."

    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]))

    gemm_kernel[grid](
        a, b, c, bias,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        M, K, N,
        fuse_relu=fuse_relu,
    )

    return c


def test_triton_torch_match():
    torch.manual_seed(0)
    a = torch.randn(1, 513, 256, device='cuda', dtype=torch.float16)
    b = torch.randn(1, 256, 513, device='cuda', dtype=torch.float16)
    b_2 = b.view(256, 513)
    bmm_output = bmm(a, b)
    mm_output = mm(a, b_2, fuse_relu=False)
    torch_output = torch.matmul(a, b)

    if torch.allclose(mm_output, torch_output, atol=1e-1):
        print("Triton and PyTorch outputs match!")
    else:
        print("Triton and PyTorch outputs do not match.")
    print(f"triton_output={mm_output}")
    print(f"torch_output={torch_output}")


if __name__ == '__main__':
    test_triton_torch_match()
    print("simple assert success")
