"""Similarity-preserving distillation from backbone to hash space (P0)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityDistillationLoss(nn.Module):
    """Transfer backbone's pairwise similarity structure to hash codes.

    The backbone (e.g. SigLIP2) has excellent cross-modal alignment from
    pre-training on billions of pairs. This loss preserves that ranking
    in the hash space via KL divergence on softmax-ed similarity matrices.

    L = (KL(student_i2t || teacher_i2t) + KL(student_t2i || teacher_t2i)) / 2

    where:
        teacher_sim = softmax(norm(backbone_img) @ norm(backbone_txt).T / tau_t)
        student_sim = log_softmax(norm(hash_img) @ norm(hash_txt).T / tau_s)
    """

    def __init__(
        self, teacher_temp: float = 0.1, student_temp: float = 0.05,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(
        self,
        backbone_img: torch.Tensor,
        backbone_txt: torch.Tensor,
        hash_img: torch.Tensor,
        hash_txt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            backbone_img: (B, D_backbone) raw backbone image embeddings.
            backbone_txt: (B, D_backbone) raw backbone text embeddings.
            hash_img: (B, D_hash) continuous hash codes for images.
            hash_txt: (B, D_hash) continuous hash codes for text.

        Returns:
            Scalar KL divergence loss.
        """
        # Teacher: backbone similarity (no gradient — teacher signal)
        with torch.no_grad():
            t_img = F.normalize(backbone_img.detach(), dim=-1)
            t_txt = F.normalize(backbone_txt.detach(), dim=-1)
            teacher_sim = t_img @ t_txt.T / self.teacher_temp
            teacher_prob_i2t = F.softmax(teacher_sim, dim=-1)
            teacher_prob_t2i = F.softmax(teacher_sim.T, dim=-1)

        # Student: hash similarity
        s_img = F.normalize(hash_img, dim=-1)
        s_txt = F.normalize(hash_txt, dim=-1)
        student_sim = s_img @ s_txt.T / self.student_temp

        student_log_i2t = F.log_softmax(student_sim, dim=-1)
        student_log_t2i = F.log_softmax(student_sim.T, dim=-1)

        # Symmetric KL divergence
        loss_i2t = F.kl_div(student_log_i2t, teacher_prob_i2t, reduction="batchmean")
        loss_t2i = F.kl_div(student_log_t2i, teacher_prob_t2i, reduction="batchmean")

        return (loss_i2t + loss_t2i) / 2.0
