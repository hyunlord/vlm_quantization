from __future__ import annotations

import torch
import torch.nn as nn


class BitBalanceLoss(nn.Module):
    """Encourages balanced, decorrelated hash bits for maximum information capacity.

    Balance:       ||mean(H, dim=0)||^2  — each bit ~50% +1/-1
    Decorrelation: ||H^T H / B - I||_F^2 — bits are independent
    """

    def __init__(self, hash_dim: int = 64):
        super().__init__()
        self.hash_dim = hash_dim

    def forward(self, hash_codes: torch.Tensor) -> torch.Tensor:
        B = hash_codes.size(0)

        # Balance: mean of each bit across batch should be 0
        balance = hash_codes.mean(dim=0).pow(2).mean()

        # Decorrelation: covariance should be identity
        cov = hash_codes.t() @ hash_codes / B
        identity = torch.eye(self.hash_dim, device=hash_codes.device)
        decorr = (cov - identity).pow(2).mean()

        return balance + decorr
