from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss for cross-modal hashing.

    Paired image-text samples (diagonal) are positives.
    All other combinations in the batch are negatives.

    L = -(log softmax(sim(i, t+) / tau))  averaged over both directions.

    Also usable for intra-modal pairs (I2I, T2T) by passing
    the same modality for both arguments.

    Focal weighting (optional): when ``focal_gamma > 0`` the per-sample
    cross-entropy is down-weighted for easy (high-confidence) positives via
    the focal factor ``(1 - p) ** gamma``, focusing learning on hard pairs.
    With ``focal_gamma == 0`` (default) the loss reduces *exactly* to the
    standard symmetric InfoNCE.
    """

    def __init__(self, temperature: float = 0.07, focal_gamma: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.focal_gamma = focal_gamma

    def _directional_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy for one direction, with optional focal weighting."""
        if self.focal_gamma == 0.0:
            return F.cross_entropy(logits, labels)
        ce = F.cross_entropy(logits, labels, reduction="none")  # (B,)
        p = torch.exp(-ce)  # probability assigned to the correct class
        return ((1.0 - p) ** self.focal_gamma * ce).mean()

    def forward(
        self,
        hash_a: torch.Tensor,
        hash_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hash_a: (B, D) continuous hash codes from modality A
            hash_b: (B, D) continuous hash codes from modality B
        """
        hash_a = F.normalize(hash_a, dim=-1)
        hash_b = F.normalize(hash_b, dim=-1)

        # (B, B) similarity matrix
        logits = hash_a @ hash_b.t() / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_a2b = self._directional_loss(logits, labels)
        loss_b2a = self._directional_loss(logits.t(), labels)
        return (loss_a2b + loss_b2a) / 2.0
