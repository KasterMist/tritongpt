import torch
from torch import Tensor

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4,
                      num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def mat_mul_kernel(
    a_ptr, b_ptr, c_ptr, # matrix a, b, c's pointers
    M, N, K, # matrix dimensions [M, K] @ [K, N]
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, # strides
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    program_size_m = tl.cdiv(M, BLOCK_SIZE_M) # to calculate C[M,N], calculate the number of programs in M 
    program_size_n = tl.cdiv(N, BLOCK_SIZE_N) # to calculate C[M,N], calculate the number of programs in N

    pid_m = pid // program_size_n # calculate the coordinate of pid in 2D dimension which focus the calculation on C[M,N]
    pid_n = pid % program_size_n # calculate the coordinate of pid in 2D dimension which focus the calculation on C[M,N]

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # calculate the program offset for C[M,N] only in M dimension
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # calculate the program offset for C[M,N] only in N dimension
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak) # for this program, get the pointer of needed matrix a's part
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn) # for this program, get the pointer of needed matrix b's part

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # for this program, initialize the result matrix c's part

    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_masks = (offset_m[:, None] < M) & offset_k[None, :] < K - i * BLOCK_SIZE_K # calculate the mask for a's part
        a = tl.load(a_ptrs, mask=a_masks, other=0.0)

        b_masks = (offset_k[:, None] < K - i * BLOCK_SIZE_K) & (offset_n[None, :] < N) # calculate the mask for b's part
        b = tl.load(b_ptrs, mask=b_masks, other=0.0)

        c += tl.dot(a, b, allow_tf32=False)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn)
    c_masks = (offset_m[:, None] < M) & (offset_n[None, :] < N) # calculate the mask for c's part
    tl.store(c_ptrs, c, mask=c_masks)

def mat_mul(a, b):
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META : (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    mat_mul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)
    )
    return c

# def matmul(input1: Tensor, input2: Tensor) -> Tensor:

#     assert input1.is_cuda and input2.is_cuda, "only supported for cuda device."
#     assert input1.size(1) == input2.size(0), "the shape of two matrices are mismatched."

#     M, K = input1.shape 
#     K, N = input2.shape 

#     output = torch.empty((M, N), device=input1.device)

#     def cal_programs_shape(meta):
#         num_pid_m = triton.cdiv(M, meta["BLOCK_SIZE_M"])
#         num_pid_n = triton.cdiv(N, meta["BLOCK_SIZE_N"])
#         num_pid = num_pid_m * num_pid_n
#         return (num_pid, )  # 返回的一定要是元组, 不能是数字

#     mat_mul_kernel[cal_programs_shape](
#         input1, input2, output, M, N, K, 
#         input1.stride(0), input1.stride(1), input2.stride(0), input2.stride(1), output.stride(0), output.stride(1),
#         BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
#     )

#     return output

def test_triton_torch_match():
    torch.manual_seed(0)
    a = torch.randn(512, 512, device='cuda', dtype=torch.float32)
    b = torch.randn(512, 512, device='cuda', dtype=torch.float32)
    triton_output = mat_mul(a, b)
    torch_output = torch.matmul(a, b)

    if torch.allclose(triton_output, torch_output, atol=1e-2):
        print("Triton and PyTorch outputs match!")
    else:
        print("Triton and PyTorch outputs do not match.")
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")


if __name__ == "__main__":
    test_triton_torch_match()
    print("simple assert success")
    # configs = []
    # configs.append(
    #     triton.testing.Benchmark(
    #         x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
    #         x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
    #         line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
    #         # Possible values for `line_arg`
    #         # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
    #         line_vals=["cublas", "triton"],  # Label name for the lines
    #         line_names=["cublas", "Triton"], # Line styles
    #         styles=[("green", "-"), ("blue", "-")],
    #         ylabel="TFLOPS",  # Label name for the y-axis
    #         plot_name="matmul-performance-fp16",  # Name for the plot, used also as a file name for saving the plot.
    #         args={}
    #     )
    # )
    # @triton.testing.perf_report(configs)
    # def benchmark(M, N, K, provider):
    #     a = torch.randn((M, K), device=torch.device("cuda:0"), dtype=torch.float16)
    #     b = torch.randn((K, N), device=torch.device("cuda:0"), dtype=torch.float16)
    #     quantiles = [0.5, 0.2, 0.8]
    #     if provider == 'cublas':
    #         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    #     if provider == 'triton':
    #         ms, min_ms, max_ms = triton.testing.do_bench(lambda: mat_mul(a, b), quantiles=quantiles)
    #     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    #     return perf(ms), perf(max_ms), perf(min_ms)
    # benchmark.run(show_plots=True, print_data=True)