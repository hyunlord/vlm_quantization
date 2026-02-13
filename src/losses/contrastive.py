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
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

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

        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)
        return (loss_a2b + loss_b2a) / 2.0
