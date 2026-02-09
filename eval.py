from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datamodule import CrossModalHashDataModule
from src.models.cross_modal_hash import CrossModalHashModel
from src.utils.metrics import (
    compute_bit_entropy,
    compute_quantization_error,
    mean_average_precision,
    precision_at_k,
)


@torch.no_grad()
def encode_all(
    model: CrossModalHashModel, dataloader: DataLoader, device: torch.device
) -> dict[str, torch.Tensor]:
    """Encode entire dataset into binary hash codes."""
    image_codes, text_codes, image_ids_all = [], [], []

    for batch in tqdm(dataloader, desc="Encoding"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        img_out = model.encode_image(batch["pixel_values"])
        txt_out = model.encode_text(batch["input_ids"], batch["attention_mask"])

        image_codes.append(img_out["binary"].cpu())
        text_codes.append(txt_out["binary"].cpu())
        image_ids_all.extend(batch["image_ids"])

    return {
        "image_codes": torch.cat(image_codes, dim=0),
        "text_codes": torch.cat(text_codes, dim=0),
        "image_ids": torch.tensor(image_ids_all),
    }


def evaluate_retrieval(
    image_codes: torch.Tensor,
    text_codes: torch.Tensor,
    image_ids: torch.Tensor,
    k_values: list[int],
) -> dict[str, dict[str, float]]:
    """Evaluate 4-direction retrieval: I2T, T2I, I2I, T2T."""
    results = {}

    directions = {
        "I2T": (image_codes, text_codes, image_ids, image_ids),
        "T2I": (text_codes, image_codes, image_ids, image_ids),
        "I2I": (image_codes, image_codes, image_ids, image_ids),
        "T2T": (text_codes, text_codes, image_ids, image_ids),
    }

    for name, (query, database, q_labels, db_labels) in directions.items():
        results[name] = {}
        results[name]["mAP@50"] = mean_average_precision(
            query, database, q_labels, db_labels, k=50
        )
        for k in k_values:
            results[name][f"P@{k}"] = precision_at_k(
                query, database, q_labels, db_labels, k=k
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-modal hashing model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CrossModalHashModel.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # DataModule
    datamodule = CrossModalHashDataModule(
        data_root=cfg["data"]["data_root"],
        processor_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_text_length=cfg["data"]["max_text_length"],
        image_size=cfg["data"]["image_size"],
    )
    datamodule.setup(stage="test")

    # Encode
    encoded = encode_all(model, datamodule.test_dataloader(), device)

    # Hash quality
    all_codes = torch.cat([encoded["image_codes"], encoded["text_codes"]], dim=0)
    entropy = compute_bit_entropy(all_codes)
    print(f"\nBit Entropy: mean={entropy.mean():.4f}, min={entropy.min():.4f}")

    # Retrieval evaluation
    k_values = [1, 5, 10, 20]
    results = evaluate_retrieval(
        encoded["image_codes"],
        encoded["text_codes"],
        encoded["image_ids"],
        k_values,
    )

    # Print results table
    print(f"\n{'Direction':<8} {'mAP@50':>8}", end="")
    for k in k_values:
        print(f" {'P@'+str(k):>8}", end="")
    print()
    print("-" * (8 + 9 + 9 * len(k_values)))

    for direction, metrics in results.items():
        print(f"{direction:<8} {metrics['mAP@50']:>8.4f}", end="")
        for k in k_values:
            print(f" {metrics[f'P@{k}']:>8.4f}", end="")
        print()


if __name__ == "__main__":
    main()
