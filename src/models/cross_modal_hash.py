from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer
from src.utils.hamming import hamming_distance, to_binary_01
from src.utils.metrics import (
    compute_bit_entropy,
    compute_quantization_error,
    mean_average_precision,
    precision_at_k,
)


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
        self._val_outputs: list[dict] = []

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

    def _pool(self, outputs) -> torch.Tensor:
        """Extract pooled embedding from model output (handles Tensor or ModelOutput)."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)

    def encode_image(
        self, pixel_values: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        outputs = self.backbone.vision_model(pixel_values=pixel_values)
        return self.image_hash(self._pool(outputs))

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> list[dict[str, torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.backbone.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.text_hash(self._pool(outputs))

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

        self._val_outputs.append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        bit_list = self.hparams.bit_list

        # Collect image_ids across all batches
        all_ids: list[int] = []
        for out in self._val_outputs:
            all_ids.extend(out["image_ids"])
        labels = torch.tensor(all_ids, device=self.device)

        # Subsample for mAP efficiency (max 5000 samples)
        N = len(labels)
        max_samples = min(N, 5000)
        if N > max_samples:
            idx = torch.randperm(N, device=self.device)[:max_samples]
        else:
            idx = torch.arange(N, device=self.device)

        # Use 64-bit codes for mAP (middle resolution)
        bit_idx = bit_list.index(64) if 64 in bit_list else 0
        bit = bit_list[bit_idx]

        img_codes = torch.cat(
            [o[f"image_binary_{bit}"] for o in self._val_outputs]
        )[idx]
        txt_codes = torch.cat(
            [o[f"text_binary_{bit}"] for o in self._val_outputs]
        )[idx]
        sub_labels = labels[idx]

        # 4-direction mAP@50
        self.log("val/map_i2t", mean_average_precision(
            img_codes, txt_codes, sub_labels, sub_labels,
        ))
        self.log("val/map_t2i", mean_average_precision(
            txt_codes, img_codes, sub_labels, sub_labels,
        ))
        self.log("val/map_i2i", mean_average_precision(
            img_codes, img_codes, sub_labels, sub_labels,
        ))
        self.log("val/map_t2t", mean_average_precision(
            txt_codes, txt_codes, sub_labels, sub_labels,
        ))

        # P@K (I2T direction — primary cross-modal metric)
        self.log("val/p1", precision_at_k(
            img_codes, txt_codes, sub_labels, sub_labels, k=1,
        ))
        self.log("val/p5", precision_at_k(
            img_codes, txt_codes, sub_labels, sub_labels, k=5,
        ))
        self.log("val/p10", precision_at_k(
            img_codes, txt_codes, sub_labels, sub_labels, k=10,
        ))

        # --- Hash Analysis for monitoring dashboard ---
        hash_analysis: dict = {}

        # 1) Per-bit activation rates for each bit level
        for k, b in enumerate(bit_list):
            all_img = torch.cat(
                [o[f"image_binary_{b}"] for o in self._val_outputs]
            )
            all_txt = torch.cat(
                [o[f"text_binary_{b}"] for o in self._val_outputs]
            )
            combined = torch.cat([all_img, all_txt], dim=0)
            activation = (combined > 0).float().mean(dim=0).cpu().tolist()
            hash_analysis[f"activation_{b}"] = activation

        # 2) Fixed 8 samples for qualitative check (deterministic seed)
        n_samples = min(8, len(idx))
        gen = torch.Generator().manual_seed(42)
        sample_perm = torch.randperm(len(idx), generator=gen)[:n_samples]

        sample_image_ids = [all_ids[idx[j].item()] for j in sample_perm]
        sample_img = img_codes[sample_perm]
        sample_txt = txt_codes[sample_perm]

        dist = hamming_distance(sample_img, sample_txt)
        similarity = 1.0 - dist.float() / float(bit)

        hash_analysis["sample_image_ids"] = [int(x) for x in sample_image_ids]
        hash_analysis["sample_img_codes"] = (
            to_binary_01(sample_img).cpu().tolist()
        )
        hash_analysis["sample_txt_codes"] = (
            to_binary_01(sample_txt).cpu().tolist()
        )
        hash_analysis["similarity_matrix"] = similarity.cpu().tolist()
        hash_analysis["bit"] = bit

        self._hash_analysis = hash_analysis

        self._val_outputs.clear()

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
