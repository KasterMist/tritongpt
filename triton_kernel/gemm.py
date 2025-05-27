import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, 
                      num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
        #               num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
        #               num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
        #               num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
        #               num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
        #               num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
        #               num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, # matrix a, b, c's pointers
    M, N, K, # matrix dimensions [M, K] @ [K, N]
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, # strides
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
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
    
    c_ptrs = c_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    c_masks = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_masks)


def matmul(a, b):
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META : (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)
    )
    return c

def test_triton_torch_match():
    torch.manual_seed(0)
    a = torch.randn(512, 512, device='cuda', dtype=torch.float32)
    b = torch.randn(512, 512, device='cuda', dtype=torch.float32)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    if torch.allclose(triton_output, torch_output, atol=1e-2):
        print("Triton and PyTorch outputs match!")
    else:
        print("Triton and PyTorch outputs do not match.")
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")

if __name__ == '__main__':
    test_triton_torch_match()
    print("simple assert success")
