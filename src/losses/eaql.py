from __future__ import annotations

import torch
import torch.nn as nn


class EAQLLoss(nn.Module):
    """Equilibrium-Aware Quantization Loss.

    Uses EMA-tracked per-bit importance to weight the quantization error,
    focusing optimization on bits that are hardest to binarize.

    L = mean_over_batch( sum_over_bits( w_d * (h_d - sign(h_d))^2 ) )
    where w_d = ema_d / sum(ema)
    """

    def __init__(self, ema_decay: float = 0.99):
        super().__init__()
        self.ema_decay = ema_decay
        self.bit_importance_ema: dict[int, torch.Tensor] = {}

    def forward(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous: (B, D) continuous hash codes (e.g. tanh output)

        Returns:
            Scalar EAQL loss.
        """
        with torch.no_grad():
            sign_target = torch.sign(continuous.detach())

        quant_error = (continuous - sign_target) ** 2  # (B, D)
        bitwise_error = quant_error.mean(dim=0)  # (D,)

        bit = continuous.size(1)
        if bit not in self.bit_importance_ema:
            self.bit_importance_ema[bit] = bitwise_error.detach().clone()
        else:
            self.bit_importance_ema[bit] = (
                self.ema_decay * self.bit_importance_ema[bit]
                + (1 - self.ema_decay) * bitwise_error.detach()
            )

        ema = self.bit_importance_ema[bit]
        weights = ema / (ema.sum() + 1e-6)  # normalize to sum=1

        weighted_error = (quant_error * weights).sum(dim=1)  # (B,)
        return weighted_error.mean()
