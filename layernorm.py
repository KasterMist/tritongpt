import torch
import torch.nn as nn
from torch.autograd import Function

from .triton_kernels.layernorm import layernorm_fwd, layernorm_bwd_da_db, layernorm_bwd_dx

class LayerNormFunction(Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def forward(ctx, input, alpha, beta, eps) -> torch.Tensor:  # ctx is context which is used to store information for forward and backward pass
        x_ln = layernorm_fwd(input, alpha, beta, eps)
        ctx.save_for_backward(input, alpha, beta)
        ctx.eps = eps

        return x_ln
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        input, alpha, beta = ctx.saved_tensors

        grad_alpha, grad_beta = layernorm_bwd_da_db(grad_output, input, ctx.eps)
        grad_input = layernorm_bwd_dx(grad_output, input, alpha, ctx.eps)

        return grad_input, grad_alpha, grad_beta, None

        

    
class LayerNorm(nn.Module):
    """
    Triton Layer Normalization
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(input, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )   


    


