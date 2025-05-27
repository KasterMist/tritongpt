import torch
import triton
import triton.language as tl

@triton.jit
def relu_mask_grad_kernel(
    grad_ptr,                   # gradient pointer
    out_act_ptr,                # output activation pointer
    batch_stride,               # stride for batch dimension
    sequence_stride,            # stride for sequence dimension
    numel,                      # num of effective elements for this block
    BLOCK_SIZE: tl.constexpr
):
    batch_pid = tl.program_id(axis=0)
    sequence_pid = tl.program_id(axis=1)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    grad_start_ptr = grad_ptr + batch_pid * batch_stride + sequence_pid * sequence_stride
    out_act_start_ptr = out_act_ptr + batch_pid * batch_stride + sequence_pid * sequence_stride

    grads = tl.load(grad_start_ptr + offsets, mask=mask)
    acts = tl.load(out_act_start_ptr + offsets, mask=mask)

    grads = tl.where(acts > 0.0, grads, 0.0)

    tl.store(grad_start_ptr + offsets, grads.to(tl.float16), mask=mask)



def relu_mask_grad(grad: torch.Tensor, out_act: torch.Tensor) -> torch.Tensor:
    batch_num, sequence_num, numel = grad.shape
    grid = (batch_num, sequence_num)

    num_warps = 4 if numel <= 2048 else 8

    relu_mask_grad_kernel[grid](
        grad,
        out_act,
        grad.stride(0),
        grad.stride(1),
        numel,
        BLOCK_SIZE=triton.next_power_of_2(numel),
        num_warps=num_warps
    )


if __name__ == '__main__':
    out_act = torch.randn((1, 512, 512), device='cuda', dtype=torch.float32)
    grad = torch.randn((1, 512, 512), device='cuda', dtype=torch.float16)
    grad_compare = grad.detach().clone()
    grad_compare = torch.where(out_act <= 0, 
                               torch.zeros_like(out_act, dtype=torch.float16), 
                               grad_compare.to(dtype=torch.float16))
    relu_mask_grad(grad, out_act)

    print("grad: ", grad)
    print('grad compare: ', grad_compare)
    assert torch.allclose(grad, grad_compare)
    print("simple assert success")
    
    
    