"""Rebuild the frozen-backbone embedding caches consumed by the head-only trainers.

Outputs (same layout the trainers expect):
  /tmp/emb_aug.pt   {"train": {"clean","weak","strong","txt": (113287, 1152) fp32, "ids": (113287,)},
                     "meta": {...}}
  /tmp/emb_cache.pt {"test":  {"img","txt": (5000, 1152) fp32, "ids": (5000,), "captions": list[str]},
                     "meta": {...}}

Conventions (kept identical to the original caches unless noted):
  - Backbone google/siglip2-so400m-patch14-384, eval, bf16 inference; features are the RAW
    (un-normalized) pooler outputs, cast to fp32. Trainers L2-normalize with NORM_IN=1.
    Encoding mirrors web/common.py: vision_model(...).pooler_output / text_model(...).pooler_output.
  - Text: `.lower()` (so400m text tower expects lowercase; commits 86a2104 / 1002b11), then
    padding="max_length", max_length=64, truncation=True.
    NOTE: the previous emb_cache was encoded in original case (PAPER_LOWERCASE_FIX.md: cos 0.9999
    vs orig-case), so caches built here are NOT byte-compatible with the old ones.
  - Data: Karpathy split from data/coco/dataset_coco.json. train = train+restval (113,287), test = 5,000,
    both in JSON order. One caption per image: sentences[0] (first caption).
  - Views: clean = A.Resize(384) (as in src/data/karpathy.py); weak = get_train_transforms();
    strong = get_consistency_augmentation() (src/data/transforms.py). Each view is drawn ONCE with
    a per-sample seed derived from --seed (42): weak seed = seed*10_000_000 + 2*idx,
    strong seed = seed*10_000_000 + 2*idx + 1 — deterministic regardless of worker scheduling.

Run on DGX:
  .venv/bin/python scripts/build_emb_cache.py [--workers 16] [--bs 64]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.data.transforms import get_consistency_augmentation, get_train_transforms  # noqa: E402

MODEL = "google/siglip2-so400m-patch14-384"
IMAGE_SIZE = 384


def load_split(karpathy_json: Path, split: str) -> list[dict]:
    data = json.load(open(karpathy_json))
    target = {"train", "restval"} if split == "train" else {split}
    return [e for e in data["images"] if e["split"] in target]


class Views(Dataset):
    def __init__(self, entries, data_root: Path, processor, augment: bool, seed: int):
        self.entries, self.root, self.proc, self.augment, self.seed = entries, data_root, processor, augment, seed
        self.resize = A.Resize(IMAGE_SIZE, IMAGE_SIZE)
        self.weak, self.strong = get_train_transforms(IMAGE_SIZE), get_consistency_augmentation(IMAGE_SIZE)

    def __len__(self):
        return len(self.entries)

    def _px(self, arr: np.ndarray) -> torch.Tensor:
        return self.proc(images=Image.fromarray(arr), return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, i):
        e = self.entries[i]
        img = np.array(Image.open(self.root / e["filepath"] / e["filename"]).convert("RGB"))
        out = [self._px(self.resize(image=img)["image"])]
        if self.augment:
            base = self.seed * 10_000_000
            self.weak.set_random_seed(base + 2 * i)
            out.append(self._px(self.weak(image=img)["image"]))
            self.strong.set_random_seed(base + 2 * i + 1)
            out.append(self._px(self.strong(image=img)["image"]))
        return torch.stack(out)  # (V, 3, H, W)


@torch.no_grad()
def encode_images(model, ds, bs, workers, dev, tag):
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True,
                    persistent_workers=False, prefetch_factor=4 if workers else None)
    V = 3 if ds.augment else 1
    outs = [torch.empty(len(ds), 1152) for _ in range(V)]
    t0 = time.perf_counter(); n = 0
    for b, px in enumerate(dl):
        B = px.shape[0]
        flat = px.view(B * V, *px.shape[2:]).to(dev, dtype=torch.bfloat16, non_blocking=True)
        f = model.vision_model(pixel_values=flat).pooler_output.float().view(B, V, -1).cpu()
        for v in range(V):
            outs[v][n:n + B] = f[:, v]
        n += B
        if b % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  [{tag}] {n}/{len(ds)} imgs {el:.0f}s ({n / max(el, 1e-6):.1f} img/s)", flush=True)
    assert n == len(ds)
    return outs


@torch.no_grad()
def encode_texts(model, tok, texts, bs, dev):
    out = torch.empty(len(texts), 1152)
    for s in range(0, len(texts), bs):
        t = tok([x.lower() for x in texts[s:s + bs]], padding="max_length", max_length=64,
                truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        out[s:s + bs] = model.text_model(input_ids=t["input_ids"].to(dev),
                                         attention_mask=am.to(dev) if am is not None else None
                                         ).pooler_output.float().cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=f"{REPO}/data/coco")
    ap.add_argument("--karpathy-json", default=f"{REPO}/data/coco/dataset_coco.json")
    ap.add_argument("--out-aug", default="/tmp/emb_aug.pt")
    ap.add_argument("--out-cache", default="/tmp/emb_cache.pt")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per split")
    a = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    dev = "cuda"
    T0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    root = Path(a.data_root)
    meta = {"model": MODEL, "dtype": "bf16 inference, fp32 storage", "normalized": False,
            "text": "lower() + padding=max_length, max_length=64, truncation", "caption": "sentences[0]",
            "seed": a.seed, "views": "clean=Resize384, weak=get_train_transforms, strong=get_consistency_augmentation"}
    timing = {}

    for split in ("test", "train"):
        ents = load_split(Path(a.karpathy_json), split)
        if a.limit:
            ents = ents[:a.limit]
        ids = torch.tensor([e["cocoid"] for e in ents])
        caps = [e["sentences"][0]["raw"] for e in ents]
        t0 = time.perf_counter()
        views = encode_images(model, Views(ents, root, proc, split == "train", a.seed), a.bs, a.workers, dev, split)
        ti = time.perf_counter() - t0; t0 = time.perf_counter()
        txt = encode_texts(model, proc.tokenizer, caps, 256, dev)
        tt = time.perf_counter() - t0
        timing[split] = {"n": len(ents), "image_sec": round(ti, 1), "text_sec": round(tt, 1)}
        print(f"[{split}] n={len(ents)} image {ti:.0f}s text {tt:.0f}s", flush=True)
        if split == "test":
            torch.save({"test": {"img": views[0], "txt": txt, "ids": ids, "captions": caps},
                        "meta": {**meta, "split": "karpathy test"}}, a.out_cache)
        else:
            torch.save({"train": {"clean": views[0], "weak": views[1], "strong": views[2], "txt": txt, "ids": ids},
                        "meta": {**meta, "split": "karpathy train+restval"}}, a.out_aug)

    peak_gpu = torch.cuda.max_memory_allocated() / 2**30
    import resource
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB -> GiB (Linux)
    summary = {"total_sec": round(time.perf_counter() - T0, 1), "timing": timing,
               "peak_gpu_alloc_gib": round(peak_gpu, 2), "peak_main_rss_gib": round(peak_rss, 2)}
    print("BUILD_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
