from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml

from src.data.datamodule import CrossModalHashDataModule
from src.models.cross_modal_hash import CrossModalHashModel


def main():
    parser = argparse.ArgumentParser(description="Train cross-modal hashing model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # DataModule
    datamodule = CrossModalHashDataModule(
        data_root=cfg["data"]["data_root"],
        processor_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_text_length=cfg["data"]["max_text_length"],
        image_size=cfg["data"]["image_size"],
    )

    # Estimate max_steps
    # COCO train2014 has ~82K images
    steps_per_epoch = 82000 // (
        cfg["training"]["batch_size"] * cfg["training"]["accumulate_grad_batches"]
    )
    max_steps = steps_per_epoch * cfg["training"]["max_epochs"]

    # Model
    model = CrossModalHashModel(
        model_name=cfg["model"]["backbone"],
        hash_dim=cfg["model"]["hash_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        hash_lr=cfg["training"]["hash_lr"],
        backbone_lr=cfg["training"]["backbone_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=max_steps,
        freeze_backbone=cfg["model"]["freeze_backbone"],
        contrastive_weight=cfg["loss"]["contrastive_weight"],
        quantization_weight=cfg["loss"]["quantization_weight"],
        balance_weight=cfg["loss"]["balance_weight"],
        consistency_weight=cfg["loss"]["consistency_weight"],
        temperature=cfg["loss"]["temperature"],
    )

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=cfg["training"].get("checkpoint_dir", "checkpoints"),
            filename="best-{epoch}-{val/total:.4f}",
            monitor="val/total",
            mode="min",
            save_top_k=3,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Monitor callback (optional)
    monitor_cfg = cfg.get("monitor", {})
    if monitor_cfg.get("enabled", False):
        from monitor.callback import MonitorCallback

        callbacks.append(
            MonitorCallback(
                server_url=monitor_cfg["server_url"],
                log_every_n_steps=monitor_cfg.get("log_every_n_steps", 10),
            )
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
