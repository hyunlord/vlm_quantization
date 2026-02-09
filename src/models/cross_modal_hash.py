from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from src.losses.combined import CombinedHashLoss
from src.models.hash_layer import HashLayer
from src.utils.metrics import compute_bit_entropy, compute_quantization_error


class CrossModalHashModel(pl.LightningModule):
    """Cross-modal hashing model using SigLIP2 dual encoders.

    Produces binary hash codes for images and text that can be
    compared via Hamming distance (XOR + popcount).
    Supports I2T, T2I, I2I, T2T retrieval.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        hash_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        hash_lr: float = 1e-3,
        backbone_lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        freeze_backbone: bool = False,
        contrastive_weight: float = 1.0,
        quantization_weight: float = 0.1,
        balance_weight: float = 0.01,
        consistency_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.config.projection_dim

        # Separate hash layers per modality
        self.image_hash = HashLayer(embed_dim, hash_dim, hidden_dim, dropout)
        self.text_hash = HashLayer(embed_dim, hash_dim, hidden_dim, dropout)

        # Loss
        self.loss_fn = CombinedHashLoss(
            contrastive_weight=contrastive_weight,
            quantization_weight=quantization_weight,
            balance_weight=balance_weight,
            consistency_weight=consistency_weight,
            temperature=temperature,
            hash_dim=hash_dim,
        )

    def encode_image(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embeds = self.backbone.get_image_features(pixel_values=pixel_values)
        return self.image_hash(image_embeds)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        text_embeds = self.backbone.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.text_hash(text_embeds)

    def forward(self, batch: dict) -> dict:
        image_out = self.encode_image(batch["pixel_values"])
        text_out = self.encode_text(batch["input_ids"], batch["attention_mask"])

        aug_image_out = None
        if "aug_pixel_values" in batch:
            aug_image_out = self.encode_image(batch["aug_pixel_values"])

        return {
            "image": image_out,
            "text": text_out,
            "aug_image": aug_image_out,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        progress = self.global_step / max(self.hparams.max_steps, 1)

        losses = self.loss_fn(
            image_continuous=outputs["image"]["continuous"],
            text_continuous=outputs["text"]["continuous"],
            image_binary=outputs["image"]["binary"],
            text_binary=outputs["text"]["binary"],
            aug_image_continuous=(
                outputs["aug_image"]["continuous"]
                if outputs["aug_image"] is not None
                else None
            ),
            progress=progress,
        )

        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=(name == "total"))

        return losses["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self(batch)

        losses = self.loss_fn(
            image_continuous=outputs["image"]["continuous"],
            text_continuous=outputs["text"]["continuous"],
            image_binary=outputs["image"]["binary"],
            text_binary=outputs["text"]["binary"],
        )

        for name, value in losses.items():
            self.log(f"val/{name}", value, sync_dist=True)

        # Hash quality metrics
        all_binary = torch.cat(
            [outputs["image"]["binary"], outputs["text"]["binary"]], dim=0
        )
        entropy = compute_bit_entropy(all_binary).mean()
        self.log("val/bit_entropy", entropy, sync_dist=True)

        quant_error = (
            compute_quantization_error(
                outputs["image"]["continuous"], outputs["image"]["binary"]
            )
            + compute_quantization_error(
                outputs["text"]["continuous"], outputs["text"]["binary"]
            )
        ) / 2.0
        self.log("val/quant_error", quant_error, sync_dist=True)

        return {
            "image_binary": outputs["image"]["binary"].detach(),
            "text_binary": outputs["text"]["binary"].detach(),
            "image_ids": batch["image_ids"],
        }

    def configure_optimizers(self):
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        hash_params = list(self.image_hash.parameters()) + list(
            self.text_hash.parameters()
        )

        param_groups = [{"params": hash_params, "lr": self.hparams.hash_lr}]
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": self.hparams.backbone_lr}
            )

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.hparams.max_steps, T_mult=1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
