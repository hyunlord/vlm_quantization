from __future__ import annotations

import torch


class SignSTE(torch.autograd.Function):
    """Sign function with Straight-Through Estimator gradient.

    Forward: sign(x) -> {-1, +1}
    Backward: passes gradient through where |x| <= 1 (clipped identity).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0.0
        return grad_input


