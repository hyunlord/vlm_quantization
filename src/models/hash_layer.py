from __future__ import annotations

import torch
import torch.nn as nn


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


class HashLayer(nn.Module):
    """Projects continuous embeddings to binary hash codes.

    Architecture: input_dim -> hidden_dim (LayerNorm + GELU) -> hash_dim -> sign()

    Returns dict with:
        continuous: tanh(projection) in [-1, 1] for loss computation
        binary: sign(projection) in {-1, +1} for inference / evaluation
    """

    def __init__(
        self,
        input_dim: int,
        hash_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hash_dim = hash_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hash_dim),
        )

        # Small weights on final layer for stable tanh output
        nn.init.xavier_uniform_(self.projection[-1].weight, gain=0.1)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(embeddings)
        continuous = torch.tanh(raw)
        binary = SignSTE.apply(raw)
        return {"continuous": continuous, "binary": binary}
