import torch
from torch import nn
from torch.autograd import Function
from einops import rearrange 

from .triton_kernels.activation import relu_mask_grad
from .triton_kernels.gemm import mm, bmm, gemm
from .triton_kernels.reduction import column_base_reduction


class LinearNoBias(Function):
    """
    LinearNoBias single kernel call
    """
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return mm(input, weight)

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors  # inputs: [batch_size, seq_len, in_features], weights: [in_features, out_features]
        grad_inputs = mm(grad_output, weights.T) # grad_output: [batch_size, seq_len, out_features]
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output) # bmm: batch, in_features, seq_len] @ [batch_size, seq_len, out_features]
        grad_weights = torch.sum(grad_weights, keepdim=False, dim=(0,)) # sum the gradient for each seq in batch, whether keepdim=True or False is acceptable
        return grad_inputs, grad_weights


class LinearBias(Function):
    """
    LinearBias single kernel call
    """
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight) # do not need to store bias because bias gradient only depends on grad_output
        return gemm(input, weight, bias, False)

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def backward(ctx, grad_output): 
        inputs, weights = ctx.saved_tensors
        grad_inputs = mm(grad_output, weights.T)
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)
        grad_bias = column_base_reduction(grad_output)
        return grad_inputs, torch.sum(grad_weights, keepdim=False, dim=(0,)), grad_bias


class LinearNoBiasReLU(Function):
    """
    LinearNoBias plus ReLU into a single kernel call
    """
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def forward(ctx, input, weight):
        output_act = mm(input, weight, True)
        ctx.save_for_backward(input, weight, output_act)
        return output_act
    
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def backward(ctx, grad_output):
        inputs, weights, output_act = ctx.saved_tensors

        relu_mask_grad(grad_output, output_act)
        grad_inputs = mm(grad_output, weights.T)
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)

        return grad_inputs, torch.sum(grad_weights, keepdim=False, dim=(0))


class LinearBiasReLU(Function):
    """
    LinearBias plus ReLU into a single kernel call
    """ 

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def forward(ctx, intput, weight, bias):
        output_act = gemm(intput, weight,bias, True)
        ctx.save_for_backward(input, weight, output_act)
        return output_act

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        relu_mask_grad(grad_output, weights.T)
        grad_inputs = mm(grad_output, weights.T)
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)
        grad_bias = column_base_reduction(grad_output)

        return grad_inputs, torch.sum(grad_weights, keepdim=False, dim=(0,)), grad_bias


class Linear(nn.Module):
    """
    Triton Linear layer with optional bias and activation fuisons
    """ 
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            fuse_activation: bool = False,
            device=None,
            dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.empty([in_features, out_features], **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.fuse_activation = fuse_activation
        self.reset_parameters()

        def reset_parameters(self) -> None:
            pass

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self.bias is None:
                if self.fuse_activation:
                    return LinearNoBiasReLU.apply(input, self.weight)
                else:
                    return LinearNoBias.apply(input, self.weight)
            else:
                if self.fuse_activation:
                    return LinearBiasReLU.apply(input, self.weight, self.bias)
                else:
                    return LinearBias.apply(input, self.weight, self.bias)
        
        def extra_repr(self) -> str:
            return "in_features={}, out_features={}, bias={}".format(
                self.in_features, self.out_features, self.bias is not None
            )


