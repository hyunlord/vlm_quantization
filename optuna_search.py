from __future__ import annotations

import argparse
import copy
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
import yaml

from src.data.datamodule import CrossModalHashDataModule
from src.models.cross_modal_hash import CrossModalHashModel
from src.utils.gpu_config import auto_configure

# Mapping: Optuna param name → config YAML path
PARAM_MAP = {
    "hidden_dim": ("model", "hidden_dim"),
    "ortho_weight": ("loss", "ortho_weight"),
    "lcs_weight": ("loss", "lcs_weight"),
    "consistency_weight": ("loss", "consistency_weight"),
    "quantization_weight": ("loss", "quantization_weight"),
    "balance_weight": ("loss", "balance_weight"),
    "temperature": ("loss", "temperature"),
    "hash_lr": ("training", "hash_lr"),
    "backbone_lr": ("training", "backbone_lr"),
    # P0: Distillation
    "distillation_weight": ("loss", "distillation_weight"),
    "distillation_teacher_temp": ("loss", "distillation_teacher_temp"),
    "distillation_student_temp": ("loss", "distillation_student_temp"),
    # P3: Focal InfoNCE
    "focal_gamma": ("loss", "focal_gamma"),
    # P5: Text augmentation
    "text_dropout_prob": ("data", "text_dropout_prob"),
}


def export_best_config(
    study: optuna.Study, base_cfg: dict, output_path: str | Path,
) -> Path:
    """Merge best trial params into base config and save as YAML."""
    best = study.best_trial
    cfg = copy.deepcopy(base_cfg)

    for param_name, (section, key) in PARAM_MAP.items():
        if param_name in best.params:
            value = best.params[param_name]
            if isinstance(value, float):
                value = round(value, 6)
            cfg[section][key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# Auto-generated from Optuna study: {study.study_name}\n")
        f.write(f"# Best trial #{best.number} — val/total: {best.value:.4f}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return output_path


def objective(
    trial: optuna.Trial, base_cfg: dict, search_epochs: int = 5,
    subset_ratio: float = 0.1,
) -> float:
    """Single Optuna trial: train for a few epochs and return validation loss."""
    cfg = copy.deepcopy(base_cfg)

    # Hyperparameters to search
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
    ortho_weight = trial.suggest_float("ortho_weight", 0.0, 0.3)
    lcs_weight = trial.suggest_float("lcs_weight", 0.0, 2.0)
    consistency_weight = trial.suggest_float("consistency_weight", 0.0, 0.5)
    quantization_weight = trial.suggest_float("quantization_weight", 0.0, 0.3)
    balance_weight = trial.suggest_float("balance_weight", 0.0, 0.05)
    temperature = trial.suggest_float("temperature", 0.03, 0.15)
    hash_lr = trial.suggest_float("hash_lr", 1e-4, 5e-3, log=True)
    backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True)

    # P0: Distillation
    distillation_weight = trial.suggest_float("distillation_weight", 0.1, 3.0)
    distillation_teacher_temp = trial.suggest_float(
        "distillation_teacher_temp", 0.05, 0.3,
    )
    distillation_student_temp = trial.suggest_float(
        "distillation_student_temp", 0.01, 0.15,
    )
    # P3: Focal InfoNCE
    focal_gamma = trial.suggest_float("focal_gamma", 0.0, 4.0)
    # P5: Text augmentation
    text_dropout_prob = trial.suggest_float("text_dropout_prob", 0.0, 0.3)

    # Fixed parameters
    bit_list = cfg["model"]["bit_list"]

    # Resolve "auto" batch_size using GPU auto-configuration
    if cfg["training"].get("batch_size") == "auto":
        gpu_cfg = auto_configure(
            freeze_backbone=cfg["model"].get("freeze_backbone", True),
        )
        cfg["training"]["batch_size"] = gpu_cfg["batch_size"]
        cfg["training"]["accumulate_grad_batches"] = gpu_cfg["accumulate_grad_batches"]
        cfg["data"]["num_workers"] = gpu_cfg["num_workers"]

    pl.seed_everything(42, workers=True)

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
        subset_ratio=subset_ratio,
        num_captions=cfg["data"].get("num_captions", 1),
        text_dropout_prob=text_dropout_prob,
    )

    # Estimate max_steps for search
    num_train_images = 113000 if karpathy_json else 82000

    # Add extra dataset sizes (count JSONL lines)
    for ds_cfg in extra_datasets or []:
        jsonl_path = ds_cfg["jsonl_path"]
        try:
            with open(jsonl_path) as f:
                num_train_images += sum(1 for line in f if line.strip())
        except FileNotFoundError:
            pass

    num_train_images = max(1, int(num_train_images * subset_ratio))
    steps_per_epoch = num_train_images // (
        cfg["training"]["batch_size"] * cfg["training"]["accumulate_grad_batches"]
    )
    max_steps = steps_per_epoch * search_epochs

    # Model with trial hyperparameters
    model = CrossModalHashModel(
        model_name=cfg["model"]["backbone"],
        bit_list=bit_list,
        hidden_dim=hidden_dim,
        dropout=cfg["model"]["dropout"],
        hash_lr=hash_lr,
        backbone_lr=backbone_lr,
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=max_steps,
        freeze_backbone=cfg["model"]["freeze_backbone"],
        contrastive_weight=1.0,  # fixed anchor
        ortho_weight=ortho_weight,
        quantization_weight=quantization_weight,
        balance_weight=balance_weight,
        consistency_weight=consistency_weight,
        lcs_weight=lcs_weight,
        temperature=temperature,
        ema_decay=cfg["loss"]["ema_decay"],
        # P0: Distillation
        distillation_weight=distillation_weight,
        distillation_teacher_temp=distillation_teacher_temp,
        distillation_student_temp=distillation_student_temp,
        # P3: Focal InfoNCE
        focal_gamma=focal_gamma,
        # P4: Learnable temperature (use config default for search)
        learnable_temp=cfg["loss"].get("learnable_temp", False),
        # P2: LoRA (use config default for search)
        use_lora=cfg["model"].get("use_lora", False),
        lora_rank=cfg["model"].get("lora_rank", 8),
        lora_alpha=cfg["model"].get("lora_alpha", 16),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
    )

    # Pruning callback
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val/total"
    )

    trainer = pl.Trainer(
        max_epochs=search_epochs,
        val_check_interval=cfg["training"].get("val_check_interval", 0.5),
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["training"]["accumulate_grad_batches"],
        callbacks=[pruning_callback],
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    # Log all metrics as user attributes
    metrics = trainer.callback_metrics
    for name, tensor in metrics.items():
        trial.set_user_attr(name, float(tensor))

    return float(metrics["val/total"])


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument(
        "--config", type=str, default="configs/colab.yaml", help="Base config"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials"
    )
    parser.add_argument(
        "--study-name", type=str, default="cross_modal_hash_opt"
    )
    parser.add_argument(
        "--storage", type=str, default="sqlite:///optuna_results.db"
    )
    parser.add_argument(
        "--search-epochs", type=int, default=5,
        help="Number of training epochs per trial (default: 5)",
    )
    parser.add_argument(
        "--subset-ratio", type=float, default=0.1,
        help="Fraction of training data per trial (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--export-config", type=str, default=None,
        help="Output path for best config YAML (default: configs/best_<study>.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial, cfg,
            search_epochs=args.search_epochs,
            subset_ratio=args.subset_ratio,
        ),
        n_trials=args.n_trials,
    )

    # Print results
    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  Value (val/total): {best.value:.4f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    if best.user_attrs:
        print("  Metrics:")
        for key, value in best.user_attrs.items():
            print(f"    {key}: {value:.4f}")

    # Export best config YAML
    export_path = args.export_config or f"configs/best_{args.study_name}.yaml"
    out = export_best_config(study, cfg, export_path)
    print(f"\n  Best config exported → {out}")
    print(f"  Retrain: python train.py --config {out}")


if __name__ == "__main__":
    main()
