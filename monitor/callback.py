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
        server_url: str = "http://localhost:8000",
        log_every_n_steps: int = 10,
    ):
        self.server_url = server_url.rstrip("/")
        self.log_every_n_steps = log_every_n_steps
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
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "loss_total": self._to_float(logged.get("train/total")),
            "loss_contrastive": self._to_float(logged.get("train/contrastive")),
            "loss_quantization": self._to_float(logged.get("train/eaql")),
            "loss_balance": self._to_float(logged.get("train/balance")),
            "loss_consistency": self._to_float(logged.get("train/consistency")),
            "loss_ortho": self._to_float(logged.get("train/ortho")),
            "loss_lcs": self._to_float(logged.get("train/lcs")),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
        })

        # Update step progress in dashboard header
        self._post("/api/training/status", {
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

        eval_data = {"epoch": trainer.current_epoch}

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

        # Precision@K
        eval_data["p1"] = self._to_float(logged.get("val/p1"), None)
        eval_data["p5"] = self._to_float(logged.get("val/p5"), None)
        eval_data["p10"] = self._to_float(logged.get("val/p10"), None)

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
                self._post("/api/metrics/hash_analysis", {
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
                })
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

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._post("/api/training/status", {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "total_epochs": trainer.max_epochs or 0,
            "total_steps": trainer.global_step,
            "is_training": False,
        })
        self._client.close()
