from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalOrthoHashLoss(nn.Module):
    """Cross-modal orthogonal hash loss with optional margin.

    In a paired batch of (image, text), the diagonal entries are
    positive pairs (same content) and off-diagonal entries are negatives.

    Pushes:
        - Matched pairs (diagonal): cosine similarity -> 1
        - Unmatched pairs (off-diagonal): cosine similarity -> 0
          (with margin: no penalty if |sim| < margin)
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, image_hash: torch.Tensor, text_hash: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_hash: (B, D) image hash codes
            text_hash:  (B, D) text hash codes

        Returns:
            Scalar ortho hash loss.
        """
        img_norm = F.normalize(image_hash, p=2, dim=1)
        txt_norm = F.normalize(text_hash, p=2, dim=1)

        sim = img_norm @ txt_norm.t()  # (B, B)
        batch_size = sim.size(0)

        identity = torch.eye(batch_size, device=sim.device)
        same_mask = identity
        diff_mask = 1.0 - identity

        loss_same = ((1.0 - sim) ** 2 * same_mask).sum() / same_mask.sum().clamp(min=1.0)

        if self.margin > 0:
            # Margin: no penalty if |sim| < margin (allow small correlations)
            loss_diff = (
                F.relu(sim.abs() - self.margin) ** 2 * diff_mask
            ).sum() / diff_mask.sum().clamp(min=1.0)
        else:
            loss_diff = (sim ** 2 * diff_mask).sum() / diff_mask.sum().clamp(min=1.0)

        return loss_same + loss_diff
