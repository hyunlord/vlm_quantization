"""Build a search index (.pt) from a trained checkpoint.

Encodes all images in a Karpathy split into hash codes + backbone embeddings,
producing a .pt file that the monitoring dashboard's search page can load.

Usage:
    python scripts/build_index.py --checkpoint checkpoints/best.ckpt
    python scripts/build_index.py --checkpoint checkpoints/best.ckpt --config configs/dgx_spark.yaml
    python scripts/build_index.py --checkpoint checkpoints/best.ckpt --split test --batch-size 64
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Allow running from project root or scripts dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cross_modal_hash import CrossModalHashModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def make_thumbnail(image: Image.Image, size: int = 128) -> str:
    """Resize PIL image to thumbnail and return base64-encoded JPEG."""
    thumb = image.copy()
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_karpathy_entries(
    karpathy_json: str | Path, split: str = "test",
) -> list[dict]:
    """Load entries from Karpathy split JSON."""
    with open(karpathy_json) as f:
        data = json.load(f)

    # train split includes restval (matches KarpathyCocoCaptionsDataset)
    target_splits = {"train", "restval"} if split == "train" else {split}
    entries = [e for e in data["images"] if e["split"] in target_splits]
    logger.info("Loaded %d entries from Karpathy '%s' split", len(entries), split)
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build search index from checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained .ckpt file",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML (for data_root / karpathy_json defaults)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="COCO data root (default: from config or data/coco)",
    )
    parser.add_argument(
        "--karpathy-json", type=str, default=None,
        help="Path to dataset_coco.json",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Karpathy split to index (default: test)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size (default: 64)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .pt path (default: <checkpoint_dir>/index_<split>.pt)",
    )
    parser.add_argument(
        "--thumbnail-size", type=int, default=128,
        help="Thumbnail pixel size (default: 128)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda, mps, cpu (default: auto-detect)",
    )
    args = parser.parse_args()

    # --------------- resolve config ---------------
    data_root = args.data_root
    karpathy_json = args.karpathy_json

    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if not data_root:
            data_root = cfg["data"]["data_root"]
        if not karpathy_json:
            karpathy_json = cfg["data"].get("karpathy_json")

    data_root = data_root or "data/coco"
    karpathy_json = karpathy_json or "data/coco/dataset_coco.json"

    # --------------- resolve device ---------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Device: %s", device)

    # --------------- load model ---------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", ckpt_path)
    model = CrossModalHashModel.load_from_checkpoint(
        str(ckpt_path), map_location="cpu", strict=False,
    )
    model.eval()
    model.to(device)

    bit_list = list(model.hparams.get("bit_list", [8, 16, 32, 48, 64, 128]))
    model_name = model.hparams.get("model_name", "google/siglip2-so400m-patch14-384")
    logger.info("Model: %s, bit_list=%s", model_name, bit_list)

    processor = AutoProcessor.from_pretrained(model_name)

    # --------------- load dataset entries ---------------
    entries = load_karpathy_entries(karpathy_json, args.split)
    if not entries:
        logger.error("No entries found for split '%s'", args.split)
        sys.exit(1)

    # --------------- output path ---------------
    # Default: save next to checkpoint so list-indices API discovers it
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ckpt_path.parent / f"index_{args.split}.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------- encode all entries ---------------
    all_image_ids: list[int] = []
    all_captions: list[str] = []
    all_thumbnails: list[str] = []
    all_backbone_img: list[torch.Tensor] = []
    all_backbone_txt: list[torch.Tensor] = []
    all_hash_img: dict[int, list[torch.Tensor]] = {b: [] for b in bit_list}
    all_hash_txt: dict[int, list[torch.Tensor]] = {b: [] for b in bit_list}

    data_root_path = Path(data_root)
    n = len(entries)
    bs = args.batch_size

    for start in tqdm(range(0, n, bs), desc="Encoding", unit="batch"):
        batch_entries = entries[start : start + bs]

        pil_images: list[Image.Image] = []
        batch_texts: list[str] = []
        batch_ids: list[int] = []
        batch_thumbs: list[str] = []

        for entry in batch_entries:
            img_path = data_root_path / entry["filepath"] / entry["filename"]
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning("Skipping %s: %s", img_path, e)
                continue

            pil_images.append(pil_img)
            batch_texts.append(entry["sentences"][0]["raw"])  # first caption
            batch_ids.append(entry["cocoid"])
            batch_thumbs.append(make_thumbnail(pil_img, args.thumbnail_size))

        if not pil_images:
            continue

        # Processor handles resizing internally
        img_inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device)

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
            # Run backbone once, reuse for both embedding and hash codes
            # (same pattern as CrossModalHashModel.validation_step)
            img_emb = model._encode_image_backbone(pixel_values)
            txt_emb = model._encode_text_backbone(input_ids, attention_mask)

            all_backbone_img.append(img_emb.cpu())
            all_backbone_txt.append(txt_emb.cpu())

            # Hash codes: adapter -> shared hash (shared bottleneck arch)
            img_adapted = model.image_adapter(img_emb)
            img_hash = model.shared_hash(img_adapted)
            for k, bit in enumerate(bit_list):
                all_hash_img[bit].append(img_hash[k]["binary"].cpu())

            txt_adapted = model.text_adapter(txt_emb)
            txt_hash = model.shared_hash(txt_adapted)
            for k, bit in enumerate(bit_list):
                all_hash_txt[bit].append(txt_hash[k]["binary"].cpu())

        all_image_ids.extend(batch_ids)
        all_captions.extend(batch_texts)
        all_thumbnails.extend(batch_thumbs)

        del pil_images  # free memory

    # --------------- assemble & save ---------------
    logger.info("Assembling index: %d items", len(all_image_ids))

    index = {
        "image_ids": all_image_ids,
        "captions": all_captions,
        "thumbnails": all_thumbnails,
        "backbone_image_emb": torch.cat(all_backbone_img, dim=0),
        "backbone_text_emb": torch.cat(all_backbone_txt, dim=0),
        "hash_image_codes": {
            b: torch.cat(all_hash_img[b], dim=0) for b in bit_list
        },
        "hash_text_codes": {
            b: torch.cat(all_hash_txt[b], dim=0) for b in bit_list
        },
    }

    torch.save(index, str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024

    logger.info("=" * 50)
    logger.info("Index saved: %s (%.1f MB)", output_path, size_mb)
    logger.info("  Items: %d", len(all_image_ids))
    logger.info("  Bit levels: %s", bit_list)
    logger.info(
        "  Backbone dim: %d", index["backbone_image_emb"].shape[1],
    )
    for bit in bit_list:
        img_shape = tuple(index["hash_image_codes"][bit].shape)
        txt_shape = tuple(index["hash_text_codes"][bit].shape)
        logger.info("  %d-bit hash: image %s, text %s", bit, img_shape, txt_shape)
    logger.info("=" * 50)
    logger.info(
        "Load in dashboard: Search page → select '%s'", output_path.name,
    )


if __name__ == "__main__":
    main()
