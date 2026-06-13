"""Build a compact *serving* index (.npz) of packed binary hash codes.

Two modes:

  1. Convert an existing dashboard index (.pt from scripts/build_index.py) into the
     packed serving format — no checkpoint or GPU needed:

        python scripts/build_serving_index.py --from-pt checkpoints/<run>/index_test.pt \
            --out indexes/serving.npz

  2. Encode a fresh image corpus from a JSONL (image_path per line):

        python scripts/build_serving_index.py --checkpoint checkpoints/<run>/best.ckpt \
            --jsonl data/corpus.jsonl --data-root data/coco --bits 16,64 \
            --out indexes/serving.npz --save-emb

The artifact holds `packed_<bit>` bitstrings per bit length plus item metadata, and
is consumed by `src/serve/api.py` (serving) and `scripts/bench_search.py` (benchmark).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serve.binary_index import pack_codes


def _save(out: Path, artifact: dict[str, np.ndarray], bits: list[int]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **artifact)
    n = len(artifact["ids"])
    size_mb = out.stat().st_size / 1024**2
    print(f"\nServing index written: {out}  ({n:,} items, {size_mb:.2f} MB)")
    for b in bits:
        print(f"  {b}-bit packed: {artifact[f'packed_{b}'].nbytes / 1024**2:.3f} MB")
    if "emb" in artifact:
        print(f"  fp32 baseline emb: {artifact['emb'].nbytes / 1024**2:.3f} MB")


def from_pt(args: argparse.Namespace) -> None:
    """Convert an existing dashboard .pt index into the packed serving format."""
    import torch

    print(f"Loading dashboard index: {args.from_pt}")
    data = torch.load(args.from_pt, map_location="cpu", weights_only=False)
    hash_codes = data["hash_image_codes"]  # dict[bit] -> (N, bit) tensor in {-1,+1}
    bits = [int(b) for b in args.bits.split(",")] if args.bits else sorted(hash_codes)
    n = len(data["image_ids"])

    artifact: dict[str, np.ndarray] = {
        "ids": np.arange(n),
        "item_ids": np.asarray(data["image_ids"]),
        "captions": np.array(data.get("captions", [""] * n), dtype=object),
    }
    for b in bits:
        artifact[f"packed_{b}"] = pack_codes(hash_codes[b].numpy())
    if args.save_emb and "backbone_image_emb" in data:
        artifact["emb"] = data["backbone_image_emb"].numpy().astype(np.float32)
    _save(Path(args.out), artifact, bits)


def _load_processor(model_name: str):
    from transformers import (
        AutoProcessor,
        GemmaTokenizer,
        SiglipImageProcessor,
        SiglipProcessor,
    )

    try:
        return AutoProcessor.from_pretrained(model_name)
    except (AttributeError, ValueError):
        return SiglipProcessor(
            image_processor=SiglipImageProcessor.from_pretrained(model_name),
            tokenizer=GemmaTokenizer.from_pretrained(model_name),
        )


def from_checkpoint(args: argparse.Namespace) -> None:
    """Encode an image corpus from a JSONL into packed codes."""
    import torch
    from PIL import Image

    from src.models.cross_modal_hash import CrossModalHashModel

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading checkpoint: {args.checkpoint}")
    model = (
        CrossModalHashModel.load_from_checkpoint(
            args.checkpoint, map_location="cpu", strict=False
        )
        .to(device)
        .eval()
    )
    bit_list = list(model.hparams.get("bit_list", [64]))
    bits = [int(b) for b in args.bits.split(",")]
    for b in bits:
        if b not in bit_list:
            raise ValueError(f"bit {b} not in model bit_list {bit_list}")
    processor = _load_processor(model.hparams.get("model_name", ""))

    entries: list[dict] = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
            if args.limit and len(entries) >= args.limit:
                break

    data_root = Path(args.data_root)
    print(f"Encoding {len(entries):,} images on {device} (bits={bits})")
    codes: dict[int, list[np.ndarray]] = {b: [] for b in bits}
    embs: list[np.ndarray] = []
    paths: list[str] = []
    captions: list[str] = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(entries), args.batch_size):
            batch = entries[start : start + args.batch_size]
            images, keep = [], []
            for e in batch:
                try:
                    images.append(
                        Image.open(data_root / e[args.image_key]).convert("RGB")
                    )
                    keep.append(e)
                except Exception as exc:
                    print(f"  skip {e.get(args.image_key)}: {exc}")
            if not images:
                continue
            pixel_values = processor(images=images, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            outs = model.encode_image(pixel_values)
            if args.save_emb:
                embs.append(model.encode_image_backbone(pixel_values).cpu().numpy())
            for b in bits:
                k = bit_list.index(b)
                codes[b].append(outs[k]["binary"].cpu().numpy().astype(np.int8))
            for e in keep:
                paths.append(e.get(args.image_key, ""))
                captions.append(e.get(args.caption_key, ""))
            done = start + len(batch)
            rate = done / (time.perf_counter() - t0 + 1e-9)
            print(f"  {done:,}/{len(entries):,}  ({rate:.1f} img/s)", end="\r")

    n = len(paths)
    artifact: dict[str, np.ndarray] = {
        "ids": np.arange(n),
        "paths": np.array(paths, dtype=object),
        "captions": np.array(captions, dtype=object),
    }
    for b in bits:
        artifact[f"packed_{b}"] = pack_codes(np.concatenate(codes[b], axis=0))
    if args.save_emb and embs:
        artifact["emb"] = np.concatenate(embs, axis=0).astype(np.float32)
    _save(Path(args.out), artifact, bits)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a packed serving index (.npz)")
    p.add_argument("--from-pt", default=None, help="convert an existing dashboard .pt")
    p.add_argument("--checkpoint", default=None, help="checkpoint for fresh encoding")
    p.add_argument("--jsonl", default=None, help="corpus JSONL (image_path per line)")
    p.add_argument("--data-root", default=".")
    p.add_argument("--out", default="indexes/serving.npz")
    p.add_argument("--bits", default="64", help="comma-separated bit lengths")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-emb", action="store_true", help="store fp32 baseline emb")
    p.add_argument("--image-key", default="image_path")
    p.add_argument("--caption-key", default="caption")
    args = p.parse_args()

    if args.from_pt:
        from_pt(args)
    elif args.checkpoint and args.jsonl:
        from_checkpoint(args)
    else:
        p.error("provide either --from-pt, or both --checkpoint and --jsonl")


if __name__ == "__main__":
    main()
