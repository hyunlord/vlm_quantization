"""Build a search index (.pt) from the Karpathy test split.

Encodes all images and captions using both backbone (1152-dim)
and hash layers (multi-bit), plus generates base64 thumbnails.

Usage:
    python scripts/build_index.py \
        --checkpoint checkpoints/.../best.ckpt \
        --data-root /data/coco \
        --karpathy-json /data/coco/dataset_coco.json \
        --output search_index.pt \
        --thumbnail-size 64
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from src.models.cross_modal_hash import CrossModalHashModel


def make_thumbnail(image_path: Path, size: int) -> str:
    """Create a base64-encoded JPEG thumbnail."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Build search index")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--karpathy-json", type=str, required=True)
    parser.add_argument("--output", type=str, default="search_index.pt")
    parser.add_argument("--thumbnail-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Load Karpathy JSON and filter to test split
    print(f"Loading Karpathy JSON: {args.karpathy_json}")
    with open(args.karpathy_json) as f:
        karpathy = json.load(f)

    entries = [e for e in karpathy["images"] if e["split"] == args.split]
    print(f"  {args.split} split: {len(entries)} images")
    del karpathy

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = CrossModalHashModel.load_from_checkpoint(
        args.checkpoint, map_location="cpu",
    )
    model.eval()

    model_name = model.hparams.get("model_name", "")
    bit_list = list(model.hparams.get("bit_list", [64]))
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"  Model: {model_name}, bit_list={bit_list}")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # Collect results
    all_backbone_img = []
    all_backbone_txt = []
    all_hash_img = {b: [] for b in bit_list}
    all_hash_txt = {b: [] for b in bit_list}
    image_ids = []
    captions = []
    image_paths = []
    thumbnails = []

    # Process in batches
    for start in tqdm(range(0, len(entries), args.batch_size), desc="Encoding"):
        batch_entries = entries[start : start + args.batch_size]

        batch_images = []
        batch_texts = []
        batch_valid = []

        for entry in batch_entries:
            img_path = data_root / entry["filepath"] / entry["filename"]
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            caption = entry["sentences"][0]["raw"]  # First caption
            batch_images.append(img)
            batch_texts.append(caption)
            batch_valid.append(entry)

        if not batch_images:
            continue

        # Process images
        img_inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device)

        # Process texts
        txt_inputs = processor.tokenizer(
            batch_texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = txt_inputs["input_ids"].to(device)
        attention_mask = txt_inputs["attention_mask"].to(device)

        with torch.no_grad():
            # Backbone embeddings
            img_emb = model.encode_image_backbone(pixel_values).cpu()
            txt_emb = model.encode_text_backbone(input_ids, attention_mask).cpu()

            # Hash codes
            img_hash = model.encode_image(pixel_values)
            txt_hash = model.encode_text(input_ids, attention_mask)

        all_backbone_img.append(img_emb)
        all_backbone_txt.append(txt_emb)

        for k, bit in enumerate(bit_list):
            all_hash_img[bit].append(img_hash[k]["binary"].cpu())
            all_hash_txt[bit].append(txt_hash[k]["binary"].cpu())

        # Metadata
        for i, entry in enumerate(batch_valid):
            image_ids.append(entry["cocoid"])
            captions.append(batch_texts[i])
            rel_path = f"{entry['filepath']}/{entry['filename']}"
            image_paths.append(rel_path)

            img_path = data_root / entry["filepath"] / entry["filename"]
            try:
                thumbnails.append(make_thumbnail(img_path, args.thumbnail_size))
            except Exception:
                thumbnails.append("")

    # Concatenate
    index_data = {
        "backbone_image_emb": torch.cat(all_backbone_img, dim=0),
        "backbone_text_emb": torch.cat(all_backbone_txt, dim=0),
        "hash_image_codes": {
            b: torch.cat(all_hash_img[b], dim=0) for b in bit_list
        },
        "hash_text_codes": {
            b: torch.cat(all_hash_txt[b], dim=0) for b in bit_list
        },
        "image_ids": image_ids,
        "captions": captions,
        "image_paths": image_paths,
        "thumbnails": thumbnails,
    }

    print(f"\nIndex built: {len(image_ids)} items")
    print(f"  Backbone emb shape: {index_data['backbone_image_emb'].shape}")
    for b in bit_list:
        print(f"  Hash {b}-bit shape: {index_data['hash_image_codes'][b].shape}")

    # Save
    torch.save(index_data, args.output)
    file_size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"  Saved to: {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
