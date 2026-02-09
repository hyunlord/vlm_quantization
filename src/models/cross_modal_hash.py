from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer
from src.utils.metrics import compute_bit_entropy, compute_quantization_error


class CrossModalHashModel(pl.LightningModule):
    """Cross-modal hashing model using SigLIP2 dual encoders.

    Produces multi-resolution binary hash codes for images and text
    via NestedHashLayer. Codes can be compared via Hamming distance
    (XOR + popcount). Supports I2T, T2I, I2I, T2T retrieval.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        bit_list: list[int] | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        hash_lr: float = 1e-3,
        backbone_lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        freeze_backbone: bool = False,
        contrastive_weight: float = 1.0,
        ortho_weight: float = 0.1,
        quantization_weight: float = 0.1,
        balance_weight: float = 0.01,
        consistency_weight: float = 0.5,
        lcs_weight: float = 0.5,
        temperature: float = 0.07,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        if bit_list is None:
            bit_list = [16, 32, 64, 128]
        self.save_hyperparameters()

        # Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # SigLIP/SigLIP2: use vision hidden size as embedding dimension
        config = self.backbone.config
        if hasattr(config, "projection_dim"):
            embed_dim = config.projection_dim
        elif hasattr(config, "vision_config"):
            embed_dim = config.vision_config.hidden_size
        else:
            embed_dim = config.hidden_size

        # Separate nested hash layers per modality
        self.image_hash = NestedHashLayer(embed_dim, hidden_dim, bit_list, dropout)
        self.text_hash = NestedHashLayer(embed_dim, hidden_dim, bit_list, dropout)

        # Loss
        self.loss_fn = CombinedHashLoss(
            bit_list=bit_list,
            contrastive_weight=contrastive_weight,
            ortho_weight=ortho_weight,
            quantization_weight=quantization_weight,
            balance_weight=balance_weight,
            consistency_weight=consistency_weight,
            lcs_weight=lcs_weight,
            temperature=temperature,
            ema_decay=ema_decay,
        )

    def encode_image(
        self, pixel_values: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        image_embeds = self.backbone.get_image_features(pixel_values=pixel_values)
        return self.image_hash(image_embeds)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> list[dict[str, torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
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
            image_outputs=outputs["image"],
            text_outputs=outputs["text"],
            aug_image_outputs=outputs["aug_image"],
            progress=progress,
        )

        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=(name == "total"))

        return losses["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self(batch)

        losses = self.loss_fn(
            image_outputs=outputs["image"],
            text_outputs=outputs["text"],
        )

        for name, value in losses.items():
            self.log(f"val/{name}", value, sync_dist=True)

        # Per-bit hash quality metrics
        bit_list = self.hparams.bit_list
        result = {"image_ids": batch["image_ids"]}

        for k, bit in enumerate(bit_list):
            img_binary = outputs["image"][k]["binary"]
            txt_binary = outputs["text"][k]["binary"]

            all_binary = torch.cat([img_binary, txt_binary], dim=0)
            entropy = compute_bit_entropy(all_binary).mean()
            self.log(f"val/{bit}_bit_entropy", entropy, sync_dist=True)

            quant_error = (
                compute_quantization_error(
                    outputs["image"][k]["continuous"], img_binary
                )
                + compute_quantization_error(
                    outputs["text"][k]["continuous"], txt_binary
                )
            ) / 2.0
            self.log(f"val/{bit}_quant_error", quant_error, sync_dist=True)

            result[f"image_binary_{bit}"] = img_binary.detach()
            result[f"text_binary_{bit}"] = txt_binary.detach()

        return result

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

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.hash_lr, self.hparams.backbone_lr]
            if backbone_params
            else [self.hparams.hash_lr],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
