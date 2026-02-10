from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

from src.data.datamodule import CrossModalHashDataModule
from src.models.cross_modal_hash import CrossModalHashModel
from src.utils.gpu_config import auto_configure


def main():
    parser = argparse.ArgumentParser(description="Train cross-modal hashing model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Auto-configure GPU-dependent parameters
    if cfg["training"].get("batch_size") == "auto":
        gpu_cfg = auto_configure(
            freeze_backbone=cfg["model"].get("freeze_backbone", True),
        )
        cfg["training"]["batch_size"] = gpu_cfg["batch_size"]
        cfg["training"]["accumulate_grad_batches"] = gpu_cfg["accumulate_grad_batches"]
        cfg["data"]["num_workers"] = gpu_cfg["num_workers"]

    # DataModule
    karpathy_json = cfg["data"].get("karpathy_json")
    datamodule = CrossModalHashDataModule(
        data_root=cfg["data"]["data_root"],
        processor_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_text_length=cfg["data"]["max_text_length"],
        image_size=cfg["data"]["image_size"],
        karpathy_json=karpathy_json,
    )

    # Estimate max_steps
    # Karpathy train+restval: ~113K images; standard COCO train2014: ~82K
    num_train_images = 113000 if karpathy_json else 82000
    steps_per_epoch = num_train_images // (
        cfg["training"]["batch_size"] * cfg["training"]["accumulate_grad_batches"]
    )
    max_steps = steps_per_epoch * cfg["training"]["max_epochs"]

    # Model
    model = CrossModalHashModel(
        model_name=cfg["model"]["backbone"],
        bit_list=cfg["model"]["bit_list"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        hash_lr=cfg["training"]["hash_lr"],
        backbone_lr=cfg["training"]["backbone_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=max_steps,
        freeze_backbone=cfg["model"]["freeze_backbone"],
        contrastive_weight=cfg["loss"]["contrastive_weight"],
        ortho_weight=cfg["loss"]["ortho_weight"],
        quantization_weight=cfg["loss"]["quantization_weight"],
        balance_weight=cfg["loss"]["balance_weight"],
        consistency_weight=cfg["loss"]["consistency_weight"],
        lcs_weight=cfg["loss"]["lcs_weight"],
        temperature=cfg["loss"]["temperature"],
        ema_decay=cfg["loss"]["ema_decay"],
    )

    # Timestamped checkpoint directory (avoid overwriting between runs)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_ckpt_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    ckpt_dir = str(Path(base_ckpt_dir) / run_id)
    print(f"  Checkpoint dir: {ckpt_dir}")

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch}-{val/total:.4f}",
            monitor="val/total",
            mode="min",
            save_top_k=3,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.EarlyStopping(
            monitor="val/total",
            mode="min",
            patience=cfg["training"].get("early_stopping_patience", 5),
            verbose=True,
        ),
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

    # Progress bar: reduce refresh rate for non-interactive environments (Colab subprocess)
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=50))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        val_check_interval=cfg["training"].get("val_check_interval", 1.0),
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Baseline validation (epoch 0, before any training)
    print("  Running baseline validation...")
    trainer.validate(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
