from __future__ import annotations

import logging

import httpx
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class MonitorCallback(pl.Callback):
    """Lightning Callback that sends metrics to the monitoring server.

    Non-blocking: failures are logged but never interrupt training.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        log_every_n_steps: int = 10,
        run_id: str = "",
    ):
        self.server_url = server_url.rstrip("/")
        self.log_every_n_steps = log_every_n_steps
        self.run_id = run_id
        self._client = httpx.Client(timeout=2.0)
        self._fail_count = 0

    def _post(self, endpoint: str, data: dict) -> None:
        try:
            self._client.post(f"{self.server_url}{endpoint}", json=data)
            self._fail_count = 0
        except Exception as e:
            self._fail_count += 1
            if self._fail_count == 1 or self._fail_count % 10 == 0:
                logger.warning(
                    "Monitor POST %s failed (%dx): %s",
                    endpoint, self._fail_count, e,
                )

    def _to_float(self, value, default=0.0) -> float | None:
        """Convert tensor/number to Python float for JSON serialization."""
        if value is None:
            return default
        return float(value)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        batches_per_epoch = len(trainer.train_dataloader)
        accum = trainer.accumulate_grad_batches
        steps_per_epoch = batches_per_epoch // accum
        self._total_steps = steps_per_epoch * (trainer.max_epochs or 0)
        self._post("/api/training/status", {
            "run_id": self.run_id,
            "epoch": 0,
            "step": 0,
            "total_epochs": trainer.max_epochs or 0,
            "total_steps": self._total_steps,
            "is_training": True,
            "config": {
                **dict(pl_module.hparams),
                "accumulate_grad_batches": accum,
                "batches_per_epoch": batches_per_epoch,
            },
        })

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        logged = trainer.callback_metrics
        self._post("/api/metrics/training", {
            "run_id": self.run_id,
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "loss_total": self._to_float(logged.get("train/total")),
            "loss_contrastive": self._to_float(logged.get("train/contrastive")),
            "loss_quantization": self._to_float(logged.get("train/eaql")),
            "loss_balance": self._to_float(logged.get("train/balance")),
            "loss_consistency": self._to_float(logged.get("train/consistency")),
            "loss_ortho": self._to_float(logged.get("train/ortho")),
            "loss_lcs": self._to_float(logged.get("train/lcs")),
            "loss_distillation": self._to_float(logged.get("train/distillation")),
            "loss_adapter_align": self._to_float(logged.get("train/adapter_align")),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "temperature": self._to_float(logged.get("train/temperature"), None),
        })

        # Update step progress in dashboard header
        self._post("/api/training/status", {
            "run_id": self.run_id,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "total_epochs": trainer.max_epochs or 0,
            "total_steps": self._total_steps,
            "is_training": True,
        })

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logged = trainer.callback_metrics
        bit_list = pl_module.hparams.get("bit_list", [64])

        eval_data = {"run_id": self.run_id, "epoch": trainer.current_epoch}

        # Per-bit hash quality metrics
        for bit in bit_list:
            eval_data[f"bit_entropy_{bit}"] = self._to_float(
                logged.get(f"val/{bit}_bit_entropy"), None
            )
            eval_data[f"quant_error_{bit}"] = self._to_float(
                logged.get(f"val/{bit}_quant_error"), None
            )

        # Aggregated hash quality (average across bit levels)
        entropy_vals = [v for k, v in eval_data.items()
                        if k.startswith("bit_entropy_") and v is not None]
        quant_vals = [v for k, v in eval_data.items()
                      if k.startswith("quant_error_") and v is not None]
        eval_data["bit_entropy"] = (
            sum(entropy_vals) / len(entropy_vals) if entropy_vals else None
        )
        eval_data["quant_error"] = (
            sum(quant_vals) / len(quant_vals) if quant_vals else None
        )

        # Retrieval metrics (mAP — I2T, T2I only)
        eval_data["map_i2t"] = self._to_float(logged.get("val/map_i2t"), None)
        eval_data["map_t2i"] = self._to_float(logged.get("val/map_t2i"), None)

        # Backbone cosine mAP baseline
        eval_data["backbone_map_i2t"] = self._to_float(
            logged.get("val/backbone_map_i2t"), None
        )
        eval_data["backbone_map_t2i"] = self._to_float(
            logged.get("val/backbone_map_t2i"), None
        )

        # Precision@K
        eval_data["p1"] = self._to_float(logged.get("val/p1"), None)
        eval_data["p5"] = self._to_float(logged.get("val/p5"), None)
        eval_data["p10"] = self._to_float(logged.get("val/p10"), None)

        # Backbone P@K baseline
        eval_data["backbone_p1"] = self._to_float(
            logged.get("val/backbone_p1"), None
        )
        eval_data["backbone_p5"] = self._to_float(
            logged.get("val/backbone_p5"), None
        )
        eval_data["backbone_p10"] = self._to_float(
            logged.get("val/backbone_p10"), None
        )

        # Validation losses (for train vs val comparison / overfitting detection)
        eval_data["step"] = trainer.global_step
        eval_data["val_loss_total"] = self._to_float(
            logged.get("val/total"), None
        )
        eval_data["val_loss_contrastive"] = self._to_float(
            logged.get("val/contrastive"), None
        )
        eval_data["val_loss_quantization"] = self._to_float(
            logged.get("val/eaql"), None
        )
        eval_data["val_loss_balance"] = self._to_float(
            logged.get("val/balance"), None
        )
        eval_data["val_loss_consistency"] = self._to_float(
            logged.get("val/consistency"), None
        )
        eval_data["val_loss_ortho"] = self._to_float(
            logged.get("val/ortho"), None
        )
        eval_data["val_loss_lcs"] = self._to_float(
            logged.get("val/lcs"), None
        )
        eval_data["val_loss_distillation"] = self._to_float(
            logged.get("val/distillation"), None
        )
        eval_data["val_loss_adapter_align"] = self._to_float(
            logged.get("val/adapter_align"), None
        )

        self._post("/api/metrics/eval", eval_data)

        # Hash analysis data (bit balance + qualitative samples)
        analysis = getattr(pl_module, "_hash_analysis", None)
        if analysis:
            try:
                val_dataset = trainer.datamodule.val_dataset
                samples = []
                for img_id in analysis["sample_image_ids"]:
                    ann_ids = val_dataset.coco.getAnnIds(imgIds=img_id)
                    anns = val_dataset.coco.loadAnns(ann_ids)
                    caption = anns[0]["caption"] if anns else ""
                    info = val_dataset.coco.imgs[img_id]
                    img_path = val_dataset.image_dir / info["file_name"]
                    thumbnail = self._make_thumbnail(img_path)
                    samples.append({
                        "image_id": img_id,
                        "caption": caption,
                        "thumbnail": thumbnail,
                    })
                # Augmentation robustness analysis
                augmentation_data = self._compute_augmentation_analysis(
                    pl_module, val_dataset, analysis,
                )

                hash_payload = {
                    "run_id": self.run_id,
                    "epoch": trainer.current_epoch,
                    "step": trainer.global_step,
                    "bit_activations": {
                        k: v for k, v in analysis.items()
                        if k.startswith("activation_")
                    },
                    "samples": samples,
                    "sample_img_codes": analysis["sample_img_codes"],
                    "sample_txt_codes": analysis["sample_txt_codes"],
                    "similarity_matrix": analysis["similarity_matrix"],
                    "bit": analysis["bit"],
                }
                if augmentation_data is not None:
                    hash_payload["augmentation"] = augmentation_data

                self._post("/api/metrics/hash_analysis", hash_payload)
            except Exception as e:
                logger.warning("Hash analysis POST failed: %s", e)

    @staticmethod
    def _make_thumbnail(path, size: int = 128) -> str:
        """Read image, resize to thumbnail, return as base64 data URI."""
        import base64
        import io

        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _make_thumbnail_from_array(arr, size: int = 128) -> str:
        """Convert numpy array to base64 thumbnail data URI."""
        import base64
        import io

        from PIL import Image

        img = Image.fromarray(arr)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _compute_augmentation_analysis(
        self,
        pl_module: pl.LightningModule,
        val_dataset,
        analysis: dict,
    ) -> dict | None:
        """Compare original vs augmented image hash codes for robustness check.

        Applies weak and strong augmentations to sample images, encodes them
        through the model, and measures Hamming similarity to the original codes.
        """
        try:
            import numpy as np
            import torch
            from PIL import Image

            from src.data.transforms import (
                get_consistency_augmentation,
                get_train_transforms,
            )
            from src.utils.hamming import to_binary_01

            device = pl_module.device
            processor = val_dataset.processor
            image_size = getattr(val_dataset, "image_size", 384)
            bit = analysis["bit"]
            bit_list = pl_module.hparams.get("bit_list", [64])
            bit_idx = bit_list.index(bit) if bit in bit_list else 0
            n_augs = 3

            weak_tf = get_train_transforms(image_size)
            strong_tf = get_consistency_augmentation(image_size)

            all_weak_pvs: list = []
            all_strong_pvs: list = []
            weak_thumbs: list[str] = []
            strong_thumbs: list[str] = []

            for img_id in analysis["sample_image_ids"]:
                info = val_dataset.coco.imgs[img_id]
                img_path = val_dataset.image_dir / info["file_name"]
                raw = np.array(Image.open(img_path).convert("RGB"))

                for j in range(n_augs):
                    w_aug = weak_tf(image=raw)["image"]
                    w_pv = processor(
                        images=Image.fromarray(w_aug), return_tensors="pt",
                    )["pixel_values"].squeeze(0)
                    all_weak_pvs.append(w_pv)
                    if j == 0:
                        weak_thumbs.append(
                            self._make_thumbnail_from_array(w_aug)
                        )

                    s_aug = strong_tf(image=raw)["image"]
                    s_pv = processor(
                        images=Image.fromarray(s_aug), return_tensors="pt",
                    )["pixel_values"].squeeze(0)
                    all_strong_pvs.append(s_pv)
                    if j == 0:
                        strong_thumbs.append(
                            self._make_thumbnail_from_array(s_aug)
                        )

            n_samples = len(analysis["sample_image_ids"])

            # Batch encode all augmented images
            with torch.no_grad():
                weak_batch = torch.stack(all_weak_pvs).to(device)
                weak_out = pl_module.encode_image(weak_batch)
                weak_codes = to_binary_01(
                    weak_out[bit_idx]["binary"]
                ).cpu()

                strong_batch = torch.stack(all_strong_pvs).to(device)
                strong_out = pl_module.encode_image(strong_batch)
                strong_codes = to_binary_01(
                    strong_out[bit_idx]["binary"]
                ).cpu()

            # Reshape: (n_samples, n_augs, bit)
            weak_codes = weak_codes.view(n_samples, n_augs, -1)
            strong_codes = strong_codes.view(n_samples, n_augs, -1)
            orig_codes = torch.tensor(
                analysis["sample_img_codes"], dtype=torch.float32,
            )

            # Per-sample similarity results
            aug_samples = []
            for i in range(n_samples):
                orig = orig_codes[i]
                w_sims = [
                    (orig == weak_codes[i, j]).float().mean().item()
                    for j in range(n_augs)
                ]
                s_sims = [
                    (orig == strong_codes[i, j]).float().mean().item()
                    for j in range(n_augs)
                ]
                aug_samples.append({
                    "image_id": int(analysis["sample_image_ids"][i]),
                    "weak_mean_sim": round(
                        sum(w_sims) / len(w_sims), 4,
                    ),
                    "weak_min_sim": round(min(w_sims), 4),
                    "strong_mean_sim": round(
                        sum(s_sims) / len(s_sims), 4,
                    ),
                    "strong_min_sim": round(min(s_sims), 4),
                    "weak_code": weak_codes[i, 0].int().tolist(),
                    "strong_code": strong_codes[i, 0].int().tolist(),
                    "weak_thumbnail": weak_thumbs[i],
                    "strong_thumbnail": strong_thumbs[i],
                })

            # Per-bit stability: how often each bit stays the same
            weak_stability = []
            strong_stability = []
            for b_pos in range(bit):
                orig_bits = orig_codes[:, b_pos]
                w_match = (
                    weak_codes[:, :, b_pos] == orig_bits.unsqueeze(1)
                ).float().mean().item()
                s_match = (
                    strong_codes[:, :, b_pos] == orig_bits.unsqueeze(1)
                ).float().mean().item()
                weak_stability.append(round(w_match, 4))
                strong_stability.append(round(s_match, 4))

            return {
                "samples": aug_samples,
                "weak_mean_overall": round(
                    sum(s["weak_mean_sim"] for s in aug_samples)
                    / len(aug_samples),
                    4,
                ),
                "strong_mean_overall": round(
                    sum(s["strong_mean_sim"] for s in aug_samples)
                    / len(aug_samples),
                    4,
                ),
                "weak_bit_stability": weak_stability,
                "strong_bit_stability": strong_stability,
                "bit": bit,
                "n_augs": n_augs,
            }
        except Exception as e:
            logger.warning("Augmentation analysis failed: %s", e)
            return None

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Called when a checkpoint is saved. Register it in the monitoring DB."""
        # Find the ModelCheckpoint callback to get the saved path
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                if cb.best_model_path:
                    self._post("/api/checkpoints/register", {
                        "run_id": self.run_id,
                        "epoch": trainer.current_epoch,
                        "step": trainer.global_step,
                        "path": cb.best_model_path,
                        "val_loss": self._to_float(
                            trainer.callback_metrics.get("val/total")
                        ),
                    })
                break

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Final checkpoint sync to catch any missed checkpoints
        self._post("/api/checkpoints/sync", {"run_id": self.run_id})

        self._post("/api/training/status", {
            "run_id": self.run_id,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "total_epochs": trainer.max_epochs or 0,
            "total_steps": trainer.global_step,
            "is_training": False,
        })
        self._client.close()
