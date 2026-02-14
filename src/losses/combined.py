from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.balance import BitBalanceLoss
from src.losses.contrastive import CrossModalContrastiveLoss
from src.losses.eaql import EAQLLoss
from src.losses.lcs import LCSSelfDistillationLoss
from src.losses.ortho_hash import CrossModalOrthoHashLoss
from src.losses.supervised import PairwiseSupervisedLoss


class CombinedHashLoss(nn.Module):
    """Multi-bit, multi-modal loss orchestrator.

    For each bit length in bit_list:
        - InfoNCE (cross-modal contrastive)
        - EAQL (adaptive quantization)
        - OrthoHash (cross-modal orthogonal)
        - BitBalance (balance + decorrelation)
        - Consistency (augmented image alignment)

    Globally across bit lengths:
        - LCS self-distillation (long -> short)

    Total = w_c * contrastive + w_o * ortho + w_q * ramp(progress) * eaql
          + w_b * balance + w_con * consistency + w_lcs * lcs
    """

    def __init__(
        self,
        bit_list: list[int],
        contrastive_weight: float = 1.0,
        ortho_weight: float = 0.1,
        quantization_weight: float = 0.1,
        balance_weight: float = 0.01,
        consistency_weight: float = 0.5,
        lcs_weight: float = 0.5,
        supervised_weight: float = 0.0,
        temperature: float = 0.07,
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
        self.supervised_weight = supervised_weight

        self.contrastive_loss = CrossModalContrastiveLoss(temperature)
        self.eaql_loss = EAQLLoss(ema_decay)
        self.ortho_loss = CrossModalOrthoHashLoss()
        self.balance_losses = nn.ModuleList(
            [BitBalanceLoss(bit) for bit in self.bit_list]
        )
        self.lcs_loss = LCSSelfDistillationLoss()
        self.supervised_loss = PairwiseSupervisedLoss(temperature)

    def forward(
        self,
        image_outputs: list[dict[str, torch.Tensor]],
        text_outputs: list[dict[str, torch.Tensor]],
        weak_image_outputs: list[dict[str, torch.Tensor]] | None = None,
        aug_image_outputs: list[dict[str, torch.Tensor]] | None = None,
        labels: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image_outputs: list of {"continuous", "binary"} per bit length (original view)
            text_outputs: same structure
            weak_image_outputs: optional weak-augmented image outputs
            aug_image_outputs: optional strong-augmented image outputs
            labels: (B, C) multi-hot label vectors for supervised loss (optional)
            progress: training progress in [0, 1] for quantization ramp-up
        """
        device = image_outputs[0]["continuous"].device
        n_bits = len(self.bit_list)

        contrastive_total = torch.tensor(0.0, device=device)
        eaql_total = torch.tensor(0.0, device=device)
        ortho_total = torch.tensor(0.0, device=device)
        balance_total = torch.tensor(0.0, device=device)
        consistency_total = torch.tensor(0.0, device=device)

        for k in range(n_bits):
            img_cont = image_outputs[k]["continuous"]
            txt_cont = text_outputs[k]["continuous"]

            # InfoNCE (cross-modal): original + augmented views ↔ text
            n_views = 1
            contrastive_total = contrastive_total + self.contrastive_loss(
                img_cont, txt_cont
            )
            if weak_image_outputs is not None:
                weak_cont = weak_image_outputs[k]["continuous"]
                contrastive_total = contrastive_total + self.contrastive_loss(
                    weak_cont, txt_cont
                )
                n_views += 1
            if aug_image_outputs is not None:
                aug_cont = aug_image_outputs[k]["continuous"]
                contrastive_total = contrastive_total + self.contrastive_loss(
                    aug_cont, txt_cont
                )
                n_views += 1

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

        # Average over bit lengths (contrastive also averaged over views)
        contrastive_total = contrastive_total / (n_bits * n_views)
        eaql_total = eaql_total / n_bits
        ortho_total = ortho_total / n_bits
        balance_total = balance_total / n_bits
        consistency_total = consistency_total / n_bits

        # LCS self-distillation (across bit lengths, both modalities)
        img_continuous_list = [out["continuous"] for out in image_outputs]
        txt_continuous_list = [out["continuous"] for out in text_outputs]
        lcs_total = (
            self.lcs_loss(img_continuous_list) + self.lcs_loss(txt_continuous_list)
        ) / 2.0

        # Supervised pairwise loss (when labels are provided and weight > 0)
        supervised_total = torch.tensor(0.0, device=device)
        if labels is not None and self.supervised_weight > 0:
            for k in range(n_bits):
                img_cont = image_outputs[k]["continuous"]
                txt_cont = text_outputs[k]["continuous"]
                supervised_total = supervised_total + self.supervised_loss(
                    img_cont, txt_cont, labels
                )
            supervised_total = supervised_total / n_bits

        # Quantization ramp-up
        quant_scale = min(1.0, progress * 2.0)

        total = (
            self.contrastive_weight * contrastive_total
            + self.ortho_weight * ortho_total
            + self.quantization_weight * quant_scale * eaql_total
            + self.balance_weight * balance_total
            + self.consistency_weight * consistency_total
            + self.lcs_weight * lcs_total
            + self.supervised_weight * supervised_total
        )

        return {
            "total": total,
            "contrastive": contrastive_total,
            "eaql": eaql_total,
            "ortho": ortho_total,
            "balance": balance_total,
            "consistency": consistency_total,
            "lcs": lcs_total,
            "supervised": supervised_total,
        }
