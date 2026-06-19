"""Zero-shot image-text retrieval benchmark for the raw SigLIP2 backbone.

Validates that our embedding pipeline reproduces the published SigLIP2 numbers
on standard benchmarks (COCO-5K, Flickr30K-1K) using the *canonical* extraction
path: official AutoProcessor (bicubic resize @ native size) + vision_model/text_model
MAP-head pooler_output + L2-normalized cosine. No hash layer is involved.

Standard 5-caption protocol:
  - T2I: each caption ranks all images; R@k = ground-truth image in top-k.
  - I2T: each image ranks all captions; R@k = any of its captions in top-k.

Published SigLIP2 so400m/14 @384 (arXiv 2502.14786, Table 1; R@1 only):
  COCO-5K   T2I R@1 = 55.8   I2T R@1 = 71.7
  Flickr-1K T2I R@1 = 85.7   I2T R@1 = 94.9

Examples:
  # COCO-5K straight from the Karpathy json (no pre-export needed)
  python scripts/zeroshot_benchmark.py --karpathy data/coco/dataset_coco.json \
      --split test --data-root data/coco --published T2I:55.8,I2T:71.7

  # Flickr30K-1K from an exported jsonl ({image_path, captions})
  python scripts/zeroshot_benchmark.py --jsonl data/flickr30k/flickr30k_test.jsonl \
      --data-root data/flickr30k --published T2I:85.7,I2T:94.9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

MODEL = "google/siglip2-so400m-patch14-384"


def _pool(out):
    """SigLIP pooled embedding = MAP-head pooler_output (matches the repo + published)."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state.mean(dim=1)


def load_processor(name: str):
    """Mirror the datamodule's robust processor loading."""
    from transformers import (
        AutoProcessor,
        GemmaTokenizer,
        SiglipImageProcessor,
        SiglipProcessor,
    )
    try:
        return AutoProcessor.from_pretrained(name)
    except (AttributeError, ValueError):
        return SiglipProcessor(
            image_processor=SiglipImageProcessor.from_pretrained(name),
            tokenizer=GemmaTokenizer.from_pretrained(name),
        )


def load_entries(args) -> list[dict]:
    """Return [{'image_path': str, 'captions': [str, ...]}], paths relative to data_root."""
    entries: list[dict] = []
    if args.karpathy:
        data = json.load(open(args.karpathy))
        for e in data["images"]:
            if e.get("split") != args.split:
                continue
            img_path = str(Path(e["filepath"]) / e["filename"]) if "filepath" in e else e["filename"]
            entries.append({
                "image_path": img_path,
                "captions": [s["raw"] for s in e["sentences"]],
            })
    elif args.jsonl:
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                caps = row["captions"] if "captions" in row else [row["caption"]]
                entries.append({"image_path": row["image_path"], "captions": caps})
    else:
        raise SystemExit("provide --karpathy or --jsonl")
    if args.limit:
        entries = entries[: args.limit]
    return entries


@torch.no_grad()
def encode(args, entries, processor, model, dev, dtype):
    """Return (img_emb [M,D], txt_emb [C,D] normalized, cap_to_img [C])."""
    # Images
    img_emb = []
    M = len(entries)
    t0 = time.perf_counter()
    for s in range(0, M, args.batch):
        batch = entries[s : s + args.batch]
        imgs = [Image.open(Path(args.data_root) / e["image_path"]).convert("RGB") for e in batch]
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype=dtype)
        img_emb.append(_pool(model.vision_model(pixel_values=px)).float().cpu())
        if (s // args.batch) % 10 == 0:
            print(f"  img {min(s+args.batch, M)}/{M} ({(s+len(batch))/(time.perf_counter()-t0+1e-9):.1f}/s)", flush=True)
    img_emb = F.normalize(torch.cat(img_emb), dim=1)

    # Captions (flattened, remember which image each belongs to)
    caps, cap_to_img = [], []
    for i, e in enumerate(entries):
        for c in e["captions"]:
            caps.append(c)
            cap_to_img.append(i)
    txt_emb = []
    tok = processor.tokenizer
    for s in range(0, len(caps), args.batch):
        t = tok(caps[s : s + args.batch], padding="max_length", max_length=64,
                truncation=True, return_tensors="pt")
        ii = t["input_ids"].to(dev)
        am = t.get("attention_mask")
        am = am.to(dev) if am is not None else None
        txt_emb.append(_pool(model.text_model(input_ids=ii, attention_mask=am)).float().cpu())
    txt_emb = F.normalize(torch.cat(txt_emb), dim=1)
    print(f"  encoded {M} images + {len(caps)} captions in {time.perf_counter()-t0:.0f}s", flush=True)
    return img_emb, txt_emb, torch.tensor(cap_to_img)


def metrics(img_emb, txt_emb, cap_to_img, ks=(1, 5, 10)):
    out = {}
    # T2I: caption -> image
    _, idx = (txt_emb @ img_emb.t()).topk(max(ks), dim=1)
    gt = cap_to_img[:, None]
    out["T2I"] = {f"R@{k}": round((idx[:, :k] == gt).any(1).float().mean().item() * 100, 2) for k in ks}
    # I2T: image -> caption
    _, idx = (img_emb @ txt_emb.t()).topk(max(ks), dim=1)
    rows = torch.arange(img_emb.size(0))[:, None]
    hit = cap_to_img[idx] == rows
    out["I2T"] = {f"R@{k}": round(hit[:, :k].any(1).float().mean().item() * 100, 2) for k in ks}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data source")
    src.add_argument("--karpathy", type=str, help="Karpathy dataset_coco.json")
    src.add_argument("--split", type=str, default="test")
    src.add_argument("--jsonl", type=str, help="generic jsonl with image_path + caption(s)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--published", type=str, default="", help="e.g. T2I:55.8,I2T:71.7")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if dev == "cpu":
        dtype = torch.float32

    entries = load_entries(args)
    print(f"benchmark: {len(entries)} images | dtype {args.dtype} | dev {dev}", flush=True)

    processor = load_processor(args.model)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(dev).eval()

    img_emb, txt_emb, cap_to_img = encode(args, entries, processor, model, dev, dtype)
    res = metrics(img_emb, txt_emb, cap_to_img)

    pub = {}
    for part in filter(None, args.published.split(",")):
        d, v = part.split(":")
        pub[d.strip()] = float(v)

    print("\n=== zero-shot retrieval (5-caption protocol) ===", flush=True)
    for d in ("T2I", "I2T"):
        line = f"{d}  " + "  ".join(f"{k}={v}" for k, v in res[d].items())
        if d in pub:
            gap = round(res[d]["R@1"] - pub[d], 2)
            verdict = "MATCH" if abs(gap) <= 1.5 else "CLOSE" if abs(gap) <= 5 else "OFF"
            line += f"   | published R@1={pub[d]}  (Δ{gap:+}  {verdict})"
        print(line, flush=True)

    if args.out:
        json.dump({"model": args.model, "dtype": args.dtype, "n_images": len(entries),
                   "results": res, "published": pub}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
