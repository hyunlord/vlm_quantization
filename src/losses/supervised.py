"""Pairwise supervised loss for multi-label cross-modal hashing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseSupervisedLoss(nn.Module):
    """Supervised cross-modal hashing loss using multi-label similarity.

    Given multi-hot label vectors, computes pairwise similarity matrix
    S[i,j] = 1 if labels_i and labels_j share any category, else 0.

    Pushes matched pairs (S=1) together and unmatched pairs (S=0) apart
    in hash code space via log-sigmoid loss.

    L = -mean(S * log(sigmoid(sim/tau)) + (1-S) * log(sigmoid(-sim/tau)))
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_hash: torch.Tensor,
        text_hash: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_hash: (B, D) continuous hash codes for images.
            text_hash: (B, D) continuous hash codes for text.
            labels: (B, C) multi-hot label vectors.

        Returns:
            Scalar pairwise supervised loss.
        """
        # Similarity matrix from multi-hot labels: S[i,j] = 1 if any overlap
        S = (labels @ labels.T > 0).float()

        # Hash similarity
        sim = image_hash @ text_hash.T / self.temperature

        # Pairwise loss (both directions)
        loss_i2t = -(
            S * F.logsigmoid(sim) + (1 - S) * F.logsigmoid(-sim)
        ).mean()
        loss_t2i = -(
            S * F.logsigmoid(sim.T) + (1 - S) * F.logsigmoid(-sim.T)
        ).mean()

        return (loss_i2t + loss_t2i) / 2.0
