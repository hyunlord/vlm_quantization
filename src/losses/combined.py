from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.contrastive import CrossModalContrastiveLoss
from src.losses.quantization import QuantizationLoss
from src.losses.balance import BitBalanceLoss


class CombinedHashLoss(nn.Module):
    """Orchestrates all loss components with configurable weights.

    Total = w_c * L_contrastive
          + w_q * L_quantization  (scaled by training progress)
          + w_b * L_balance
          + w_con * L_consistency
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        quantization_weight: float = 0.1,
        balance_weight: float = 0.01,
        consistency_weight: float = 0.5,
        temperature: float = 0.07,
        hash_dim: int = 64,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.quantization_weight = quantization_weight
        self.balance_weight = balance_weight
        self.consistency_weight = consistency_weight

        self.contrastive_loss = CrossModalContrastiveLoss(temperature)
        self.quantization_loss = QuantizationLoss()
        self.balance_loss = BitBalanceLoss(hash_dim)

    def forward(
        self,
        image_continuous: torch.Tensor,
        text_continuous: torch.Tensor,
        image_binary: torch.Tensor,
        text_binary: torch.Tensor,
        aug_image_continuous: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            progress: training progress in [0, 1] for quantization weight ramp-up.
        """
        losses: dict[str, torch.Tensor] = {}

        # Cross-modal contrastive
        losses["contrastive"] = self.contrastive_loss(
            image_continuous, text_continuous
        )

        # Quantization (both modalities), ramped up over training
        quant_scale = min(1.0, progress * 2.0)
        losses["quantization"] = (
            self.quantization_loss(image_continuous, image_binary)
            + self.quantization_loss(text_continuous, text_binary)
        ) / 2.0

        # Bit balance (joint across modalities)
        all_hashes = torch.cat([image_continuous, text_continuous], dim=0)
        losses["balance"] = self.balance_loss(all_hashes)

        # Consistency (augmented image → same hash)
        if aug_image_continuous is not None:
            losses["consistency"] = F.mse_loss(
                image_continuous, aug_image_continuous
            )
        else:
            losses["consistency"] = torch.tensor(0.0, device=image_continuous.device)

        losses["total"] = (
            self.contrastive_weight * losses["contrastive"]
            + self.quantization_weight * quant_scale * losses["quantization"]
            + self.balance_weight * losses["balance"]
            + self.consistency_weight * losses["consistency"]
        )

        return losses
