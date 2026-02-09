from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalOrthoHashLoss(nn.Module):
    """Cross-modal orthogonal hash loss.

    In a paired batch of (image, text), the diagonal entries are
    positive pairs (same content) and off-diagonal entries are negatives.

    Pushes:
        - Matched pairs (diagonal): cosine similarity -> 1
        - Unmatched pairs (off-diagonal): cosine similarity -> 0
    """

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
        loss_diff = (sim ** 2 * diff_mask).sum() / diff_mask.sum().clamp(min=1.0)

        return loss_same + loss_diff
