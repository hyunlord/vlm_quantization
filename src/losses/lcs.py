from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCSSelfDistillationLoss(nn.Module):
    """Long-to-Short Cascade Self-Distillation.

    Longer hash codes (teacher) teach shorter hash codes (student)
    to preserve pairwise similarity structure.

    For adjacent bit lengths [b_0, b_1, ..., b_n]:
        teacher = hash_codes[i+1].detach()  (longer, no gradient)
        student = hash_codes[i]             (shorter, receives gradient)
        loss = MSE(sim_student, sim_teacher)

    where sim = normalize(codes @ codes.T)
    """

    def forward(self, hash_codes_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hash_codes_list: List of (B, D_i) tensors, one per bit length,
                             sorted by ascending bit length.

        Returns:
            Scalar LCS loss (sum over adjacent pairs).
        """
        if len(hash_codes_list) < 2:
            return torch.tensor(0.0, device=hash_codes_list[0].device)

        total_loss = torch.tensor(0.0, device=hash_codes_list[0].device)

        for i in range(len(hash_codes_list) - 1):
            teacher = hash_codes_list[i + 1].detach()  # longer code
            student = hash_codes_list[i]  # shorter code

            sim_teacher = F.normalize(teacher @ teacher.t())
            sim_student = F.normalize(student @ student.t())

            total_loss = total_loss + F.mse_loss(sim_student, sim_teacher)

        return total_loss
