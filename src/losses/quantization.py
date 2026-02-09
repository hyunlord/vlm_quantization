from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizationLoss(nn.Module):
    """Minimizes the gap between continuous and binary hash codes.

    L_quant = MSE(tanh(h), sign(h).detach())

    Encourages the network to produce outputs close to {-1, +1},
    reducing information loss from binarization.
    """

    def forward(
        self,
        continuous: torch.Tensor,
        binary: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(continuous, binary.detach())
