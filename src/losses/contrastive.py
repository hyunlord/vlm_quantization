from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss for cross-modal hashing.

    Paired image-text samples (diagonal) are positives.
    All other combinations in the batch are negatives.

    Supports:
        - Learnable temperature (P4): temperature is a trainable parameter
        - Focal weighting (P3): down-weight easy negatives via (1 - p_pos)^gamma

    L = -(log softmax(sim(i, t+) / tau))  averaged over both directions.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = False,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma

        if learnable_temp:
            # Learnable log-temperature (P4)
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer(
                "log_temp", torch.log(torch.tensor(temperature))
            )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value (clamped for stability)."""
        return self.log_temp.exp().clamp(0.01, 1.0)

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

        if self.focal_gamma > 0:
            # Focal InfoNCE (P3): focus on hard negatives
            loss_a2b = self._focal_cross_entropy(logits, labels)
            loss_b2a = self._focal_cross_entropy(logits.t(), labels)
        else:
            loss_a2b = F.cross_entropy(logits, labels)
            loss_b2a = F.cross_entropy(logits.t(), labels)

        return (loss_a2b + loss_b2a) / 2.0

    def _focal_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy with focal weighting to down-weight easy samples."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - target_probs) ** self.focal_gamma

        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        return (focal_weight * ce_loss).mean()
