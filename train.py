from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Suppress noisy pynvml FutureWarning from torch.cuda (repeated per DataLoader worker)
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)

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
    extra_datasets = cfg["data"].get("extra_datasets")
    datamodule = CrossModalHashDataModule(
        data_root=cfg["data"]["data_root"],
        processor_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_text_length=cfg["data"]["max_text_length"],
        image_size=cfg["data"]["image_size"],
        karpathy_json=karpathy_json,
        extra_datasets=extra_datasets,
        num_captions=cfg["data"].get("num_captions", 1),
        text_dropout_prob=cfg["data"].get("text_dropout_prob", 0.0),
    )

    # Estimate max_steps
    # Karpathy train+restval: ~113K images; standard COCO train2014: ~82K
    num_train_images = 113000 if karpathy_json else 82000

    # Add extra dataset sizes (count JSONL lines)
    for ds_cfg in extra_datasets or []:
        jsonl_path = ds_cfg["jsonl_path"]
        try:
            with open(jsonl_path) as f:
                count = sum(1 for line in f if line.strip())
            num_train_images += count
            print(f"  Extra dataset: {jsonl_path} ({count:,} samples)")
        except FileNotFoundError:
            print(f"  Warning: {jsonl_path} not found, skipping count")

    print(f"  Total training images: {num_train_images:,}")

    steps_per_epoch = num_train_images // (
        cfg["training"]["batch_size"] * cfg["training"]["accumulate_grad_batches"]
    )
    max_steps = steps_per_epoch * cfg["training"]["max_epochs"]

    # Model
    model = CrossModalHashModel(
        model_name=cfg["model"]["backbone"],
        bit_list=cfg["model"]["bit_list"],
        hidden_dim=cfg["model"]["hidden_dim"],
        shared_dim=cfg["model"].get("shared_dim", 768),
        dropout=cfg["model"]["dropout"],
        progressive_hash=cfg["model"].get("progressive_hash", False),
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
        # P0: Backbone similarity distillation
        distillation_weight=cfg["loss"].get("distillation_weight", 1.0),
        distillation_teacher_temp=cfg["loss"].get("distillation_teacher_temp", 0.1),
        distillation_student_temp=cfg["loss"].get("distillation_student_temp", 0.05),
        # Adapter alignment loss
        adapter_align_weight=cfg["loss"].get("adapter_align_weight", 0.1),
        # P3: Focal InfoNCE
        focal_gamma=cfg["loss"].get("focal_gamma", 0.0),
        # P4: Learnable temperature
        learnable_temp=cfg["loss"].get("learnable_temp", False),
        # OrthoHash margin + two-stage quantization
        ortho_margin=cfg["loss"].get("ortho_margin", 0.0),
        quantization_start_progress=cfg["loss"].get("quantization_start_progress", 0.4),
        # P2: LoRA fine-tuning
        use_lora=cfg["model"].get("use_lora", False),
        lora_rank=cfg["model"].get("lora_rank", 8),
        lora_alpha=cfg["model"].get("lora_alpha", 16),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
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
                run_id=run_id,
            )
        )

    # Progress bar
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=1))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        val_check_interval=cfg["training"].get("val_check_interval", 0.5),
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # skip: we run full validate() below
    )

    # Baseline validation (epoch 0, before any training)
    print("  Running baseline validation...")
    trainer.validate(model, datamodule=datamodule)

    # Standalone validate() may not trigger callbacks in all Lightning versions.
    # Explicitly push baseline eval metrics to the monitor dashboard.
    if monitor_cfg.get("enabled", False):
        from monitor.callback import MonitorCallback

        for cb in trainer.callbacks:
            if isinstance(cb, MonitorCallback):
                cb.on_validation_epoch_end(trainer, model)
                break

    # Clean up GPU state before training to avoid CUDA fork deadlocks
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
