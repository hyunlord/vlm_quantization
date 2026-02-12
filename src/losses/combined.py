from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.balance import BitBalanceLoss
from src.losses.contrastive import CrossModalContrastiveLoss
from src.losses.distillation import SimilarityDistillationLoss
from src.losses.eaql import EAQLLoss
from src.losses.lcs import LCSSelfDistillationLoss
from src.losses.ortho_hash import CrossModalOrthoHashLoss


class CombinedHashLoss(nn.Module):
    """Multi-bit, multi-modal loss orchestrator.

    For each bit length in bit_list:
        - InfoNCE (cross-modal contrastive) with focal weighting (P3)
        - EAQL (adaptive quantization)
        - OrthoHash (cross-modal orthogonal, with margin)
        - BitBalance (balance + decorrelation)
        - Consistency (augmented image alignment)

    Globally across bit lengths:
        - LCS self-distillation (long -> short)
        - Backbone similarity distillation (P0)
        - Adapter alignment (cosine embedding loss)

    Quantization schedule: cosine ramp-up starting at quantization_start_progress.
    """

    def __init__(
        self,
        bit_list: list[int],
        contrastive_weight: float = 1.0,
        ortho_weight: float = 0.01,
        quantization_weight: float = 0.1,
        balance_weight: float = 0.01,
        consistency_weight: float = 0.5,
        lcs_weight: float = 0.5,
        distillation_weight: float = 1.0,
        adapter_align_weight: float = 0.1,
        temperature: float = 0.07,
        learnable_temp: bool = False,
        focal_gamma: float = 0.0,
        ortho_margin: float = 0.0,
        quantization_start_progress: float = 0.4,
        distillation_teacher_temp: float = 0.1,
        distillation_student_temp: float = 0.05,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.bit_list = sorted(bit_list)
        self.contrastive_weight = contrastive_weight
        self.ortho_weight = ortho_weight
        self.quantization_weight = quantization_weight
        self.balance_weight = balance_weight
        self.consistency_weight = consistency_weight
        self.lcs_weight = lcs_weight
        self.distillation_weight = distillation_weight
        self.adapter_align_weight = adapter_align_weight
        self.quantization_start_progress = quantization_start_progress

        self.contrastive_loss = CrossModalContrastiveLoss(
            temperature, learnable_temp=learnable_temp, focal_gamma=focal_gamma,
        )
        self.eaql_loss = EAQLLoss(ema_decay, bit_list=bit_list)
        self.ortho_loss = CrossModalOrthoHashLoss(margin=ortho_margin)
        self.balance_losses = nn.ModuleList(
            [BitBalanceLoss(bit) for bit in self.bit_list]
        )
        self.lcs_loss = LCSSelfDistillationLoss()
        self.distillation_loss = SimilarityDistillationLoss(
            teacher_temp=distillation_teacher_temp,
            student_temp=distillation_student_temp,
        )

    def _quantization_scale(self, progress: float) -> float:
        """Cosine ramp-up schedule for quantization loss."""
        start = self.quantization_start_progress
        if progress < start:
            return 0.0
        t = (progress - start) / max(1.0 - start, 1e-8)
        t = min(t, 1.0)
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def forward(
        self,
        image_outputs: list[dict[str, torch.Tensor]],
        text_outputs: list[dict[str, torch.Tensor]],
        weak_image_outputs: list[dict[str, torch.Tensor]] | None = None,
        aug_image_outputs: list[dict[str, torch.Tensor]] | None = None,
        aux_text_outputs: list[dict[str, torch.Tensor]] | None = None,
        backbone_img: torch.Tensor | None = None,
        backbone_txt: torch.Tensor | None = None,
        adapted_img: torch.Tensor | None = None,
        adapted_txt: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image_outputs: list of {"continuous", "binary"} per bit length
            text_outputs: same structure
            weak_image_outputs: optional weak-augmented image outputs
            aug_image_outputs: optional strong-augmented image outputs
            aux_text_outputs: optional auxiliary caption outputs (P1)
            backbone_img: (B, D) backbone embeddings for distillation (P0)
            backbone_txt: (B, D) backbone embeddings for distillation (P0)
            adapted_img: (B, shared_dim) adapter output for alignment loss
            adapted_txt: (B, shared_dim) adapter output for alignment loss
            progress: training progress in [0, 1]
        """
        device = image_outputs[0]["continuous"].device
        n_bits = len(self.bit_list)

        contrastive_total = torch.tensor(0.0, device=device)
        eaql_total = torch.tensor(0.0, device=device)
        ortho_total = torch.tensor(0.0, device=device)
        balance_total = torch.tensor(0.0, device=device)
        consistency_total = torch.tensor(0.0, device=device)
        distillation_total = torch.tensor(0.0, device=device)

        # Number of image views: original + optional weak + optional strong
        n_img_views = 1 + (weak_image_outputs is not None) + (aug_image_outputs is not None)
        # Number of text views: primary + optional auxiliary
        n_txt_views = 1 + (aux_text_outputs is not None)
        n_contrastive_pairs = n_img_views * n_txt_views

        for k in range(n_bits):
            img_cont = image_outputs[k]["continuous"]
            txt_cont = text_outputs[k]["continuous"]

            # InfoNCE (cross-modal): all image views x all text views
            contrastive_total = contrastive_total + self.contrastive_loss(
                img_cont, txt_cont
            )
            if aux_text_outputs is not None:
                aux_txt_cont = aux_text_outputs[k]["continuous"]
                contrastive_total = contrastive_total + self.contrastive_loss(
                    img_cont, aux_txt_cont
                )
            if weak_image_outputs is not None:
                weak_cont = weak_image_outputs[k]["continuous"]
                contrastive_total = contrastive_total + self.contrastive_loss(
                    weak_cont, txt_cont
                )
                if aux_text_outputs is not None:
                    contrastive_total = contrastive_total + self.contrastive_loss(
                        weak_cont, aux_txt_cont
                    )
            if aug_image_outputs is not None:
                aug_cont = aug_image_outputs[k]["continuous"]
                contrastive_total = contrastive_total + self.contrastive_loss(
                    aug_cont, txt_cont
                )
                if aux_text_outputs is not None:
                    contrastive_total = contrastive_total + self.contrastive_loss(
                        aug_cont, aux_txt_cont
                    )

            # EAQL (both modalities)
            eaql_total = eaql_total + (
                self.eaql_loss(img_cont) + self.eaql_loss(txt_cont)
            ) / 2.0

            # OrthoHash (cross-modal)
            ortho_total = ortho_total + self.ortho_loss(img_cont, txt_cont)

            # Balance (joint)
            all_hashes = torch.cat([img_cont, txt_cont], dim=0)
            balance_total = balance_total + self.balance_losses[k](all_hashes)

            # Consistency: hub-and-spoke (original as anchor)
            if weak_image_outputs is not None:
                consistency_total = consistency_total + F.mse_loss(
                    img_cont, weak_image_outputs[k]["continuous"]
                )
            if aug_image_outputs is not None:
                consistency_total = consistency_total + F.mse_loss(
                    img_cont, aug_image_outputs[k]["continuous"]
                )

            # Backbone similarity distillation (P0) — per bit level
            if (
                self.distillation_weight > 0
                and backbone_img is not None
                and backbone_txt is not None
            ):
                distillation_total = distillation_total + self.distillation_loss(
                    backbone_img, backbone_txt, img_cont, txt_cont,
                )

        # Average over bit lengths
        contrastive_total = contrastive_total / (n_bits * n_contrastive_pairs)
        eaql_total = eaql_total / n_bits
        ortho_total = ortho_total / n_bits
        balance_total = balance_total / n_bits
        consistency_total = consistency_total / n_bits
        if self.distillation_weight > 0:
            distillation_total = distillation_total / n_bits

        # LCS self-distillation (across bit lengths, both modalities)
        img_continuous_list = [out["continuous"] for out in image_outputs]
        txt_continuous_list = [out["continuous"] for out in text_outputs]
        lcs_total = (
            self.lcs_loss(img_continuous_list) + self.lcs_loss(txt_continuous_list)
        ) / 2.0

        # Adapter alignment loss (cosine embedding)
        adapter_align_total = torch.tensor(0.0, device=device)
        if (
            self.adapter_align_weight > 0
            and adapted_img is not None
            and adapted_txt is not None
        ):
            adapter_align_total = 1.0 - F.cosine_similarity(
                adapted_img, adapted_txt, dim=-1
            ).mean()

        # Quantization cosine ramp-up schedule
        quant_scale = self._quantization_scale(progress)

        total = (
            self.contrastive_weight * contrastive_total
            + self.ortho_weight * ortho_total
            + self.quantization_weight * quant_scale * eaql_total
            + self.balance_weight * balance_total
            + self.consistency_weight * consistency_total
            + self.lcs_weight * lcs_total
            + self.distillation_weight * distillation_total
            + self.adapter_align_weight * adapter_align_total
        )

        result = {
            "total": total,
            "contrastive": contrastive_total,
            "eaql": eaql_total,
            "ortho": ortho_total,
            "balance": balance_total,
            "consistency": consistency_total,
            "lcs": lcs_total,
        }

        # Only include optional losses in output if active
        if self.distillation_weight > 0:
            result["distillation"] = distillation_total
        if self.adapter_align_weight > 0 and adapted_img is not None:
            result["adapter_align"] = adapter_align_total

        return result
