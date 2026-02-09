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

    def _post(self, endpoint: str, data: dict) -> None:
        try:
            self._client.post(f"{self.server_url}{endpoint}", json=data)
        except Exception as e:
            logger.debug(f"Monitor send failed: {e}")

    def _to_float(self, value, default=0.0) -> float | None:
        """Convert tensor/number to Python float for JSON serialization."""
        if value is None:
            return default
        return float(value)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        steps_per_epoch = (
            len(trainer.train_dataloader)
            // trainer.accumulate_grad_batches
        )
        self._post("/api/training/status", {
            "epoch": 0,
            "step": 0,
            "total_epochs": trainer.max_epochs or 0,
            "total_steps": steps_per_epoch * (trainer.max_epochs or 0),
            "is_training": True,
            "config": dict(pl_module.hparams),
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

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logged = trainer.callback_metrics
        bit_list = pl_module.hparams.get("bit_list", [64])

        eval_data = {"epoch": trainer.current_epoch}
        for bit in bit_list:
            eval_data[f"bit_entropy_{bit}"] = self._to_float(
                logged.get(f"val/{bit}_bit_entropy"), None
            )
            eval_data[f"quant_error_{bit}"] = self._to_float(
                logged.get(f"val/{bit}_quant_error"), None
            )

        self._post("/api/metrics/eval", eval_data)

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
