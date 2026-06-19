"""Build the browser-side search artifacts from a curated COCO subset.

Reads cached SigLIP2 embeddings (demo_index.npz), encodes them with the ft113 hash
head into 1024-bit codes, and writes three artifacts the browser consumes:

  <out>/data/index.bin        flat packed codes; row i = bytes[i*128:(i+1)*128]  (headerless)
  <out>/data/meta.json        [{id, thumb, w, h, caption}] aligned 1:1 with index.bin rows
  <out>/data/index_info.json  {n, bits, code_bytes, head, norm_in, ...}  (format descriptor)
  <out>/thumbs/<id>.jpg        <=200px JPEG thumbnails

No vector DB, no per-query inference at serve time: the browser loads index.bin once
and runs Hamming search locally; query_server.py only encodes the query text.

Selection: a `seed`-seeded random subset of the corpus. Only rows whose source image
opens successfully are kept (so every meta row has a real thumbnail and index.bin and
meta.json stay perfectly aligned). Candidates are consumed in shuffled order until
`n` thumbnails are produced.

Run on DGX (checkpoint + embeddings + images live there):
  cd ~/github/vlm_quantization
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/build_index.py \
      --n 50000 --seed 42 --index /tmp/demo_index.npz \
      --image-root data/coco --out web/static
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BITS, CODE_BYTES, HEAD_PATH, Encoder

_THUMB_PX = 200
_ID_RE = re.compile(r"_0*(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


def _derive_id(path: str, row: int) -> str:
    """COCO filename -> integer image id; fall back to the corpus row index."""
    m = _ID_RE.search(path)
    return m.group(1) if m else f"row{row}"


def _make_thumb(args_tuple) -> tuple[int, str, int, int, bool]:
    """Worker: open one image, resize to <=PX, save JPEG. Returns (row, id, w, h, ok).

    With reuse=True an already-written thumbnail is accepted as-is (fast rebuilds).
    """
    row, img_path, item_id, thumbs_dir, px, reuse = args_tuple
    out = os.path.join(thumbs_dir, f"{item_id}.jpg")
    try:
        from PIL import Image

        if reuse and os.path.exists(out):
            with Image.open(out) as im:
                w, h = im.size
            return row, item_id, w, h, True
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im.thumbnail((px, px), Image.LANCZOS)
            w, h = im.size
            im.save(out, "JPEG", quality=85)
        return row, item_id, w, h, True
    except Exception:
        return row, item_id, 0, 0, False


def main() -> None:
    p = argparse.ArgumentParser(description="Build browser-side 1-bit search index")
    p.add_argument("--n", type=int, default=50000, help="target corpus size (kept rows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--index", default="/tmp/demo_index.npz", help="cached-embedding npz")
    p.add_argument("--image-root", default="data/coco", help="root for npz `paths`")
    p.add_argument("--out", default="web/static", help="static dir (data/ + thumbs/ go here)")
    p.add_argument("--head", default=HEAD_PATH, help="hash head .pt (img_h/txt_h)")
    p.add_argument("--thumb-px", type=int, default=_THUMB_PX)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    p.add_argument("--captions-json", default=None,
                   help="COCO dataset_coco.json for caption fallback "
                        "(default <image-root>/dataset_coco.json)")
    p.add_argument("--reuse-thumbs", action="store_true",
                   help="reuse existing thumbnails (skip re-encode) for fast rebuilds")
    args = p.parse_args()

    out = Path(args.out)
    data_dir = out / "data"
    thumbs_dir = out / "thumbs"
    data_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    print(f"[build] loading corpus embeddings: {args.index}", flush=True)
    ix = np.load(args.index, allow_pickle=True)
    emb_all = ix["emb"].astype(np.float32)
    paths_all = [str(x) for x in ix["paths"]]
    caps_all = [str(x) for x in ix["captions"]] if "captions" in ix.files else [""] * len(paths_all)
    n_total = len(paths_all)
    print(f"[build] corpus N={n_total:,}  emb={emb_all.shape}", flush=True)
    if args.n > n_total:
        raise SystemExit(f"--n {args.n} > corpus {n_total}")

    # COCO caption fallback: demo_index.npz often stores empty captions for COCO; the
    # real captions live in dataset_coco.json keyed by cocoid (same as demo/live_server).
    capmap: dict[int, str] = {}
    caps_json = Path(args.captions_json) if args.captions_json else (image_root / "dataset_coco.json")
    if caps_json.exists():
        for im in json.load(open(caps_json)).get("images", []):
            if im.get("sentences"):
                capmap[int(im["cocoid"])] = im["sentences"][0]["raw"]
        print(f"[build] caption fallback: {len(capmap):,} captions <- {caps_json}", flush=True)

    def _caption(row: int, item_id: str) -> str:
        c = caps_all[row].strip()
        if c:
            return c[:200]
        try:
            return capmap.get(int(item_id), "")[:200]
        except (ValueError, TypeError):
            return ""

    # Shuffled candidate order; consume until `n` thumbnails succeed (keeps alignment).
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_total)

    kept_rows: list[int] = []
    meta_by_row: dict[int, dict] = {}
    t0 = time.perf_counter()
    cursor = 0
    chunk = max(args.n // 8, 4096)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while len(kept_rows) < args.n and cursor < n_total:
            batch_idx = order[cursor:cursor + chunk]
            cursor += chunk
            jobs = []
            for r in batch_idx:
                r = int(r)
                iid = _derive_id(paths_all[r], r)
                jobs.append((r, str(image_root / paths_all[r]), iid, str(thumbs_dir),
                             args.thumb_px, args.reuse_thumbs))
            for row, iid, w, h, ok in ex.map(_make_thumb, jobs, chunksize=64):
                if ok:
                    meta_by_row[row] = {"id": iid, "thumb": f"thumbs/{iid}.jpg",
                                        "w": w, "h": h, "caption": _caption(row, iid)}
            # preserve shuffled order among the successes in this batch
            for r in batch_idx:
                r = int(r)
                if r in meta_by_row and r not in kept_rows:
                    kept_rows.append(r)
                    if len(kept_rows) >= args.n:
                        break
            rate = len(kept_rows) / (time.perf_counter() - t0 + 1e-9)
            print(f"[build] thumbs {len(kept_rows):,}/{args.n:,} "
                  f"(scanned {cursor:,}, {rate:.0f}/s)", flush=True)

    if len(kept_rows) < args.n:
        print(f"[build] WARNING: only {len(kept_rows):,}/{args.n:,} images opened "
              f"(corpus exhausted) — building with what we have.", flush=True)

    kept_rows = kept_rows[:args.n]
    meta = [meta_by_row[r] for r in kept_rows]
    n = len(kept_rows)

    # Encode the kept rows -> packed 1024-bit codes (row order == meta order).
    print(f"[build] encoding {n:,} rows with head {args.head}", flush=True)
    enc = Encoder(head_path=args.head)
    emb_sub = emb_all[np.asarray(kept_rows)]
    packed = enc.image_codes_packed(emb_sub)  # (n, CODE_BYTES) uint8
    assert packed.shape == (n, CODE_BYTES), f"bad packed shape {packed.shape}"
    assert packed.dtype == np.uint8 and packed.flags["C_CONTIGUOUS"]

    # Write artifacts.
    (data_dir / "index.bin").write_bytes(packed.tobytes())  # row-major flat bytes
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    info = {"n": n, "bits": CODE_BITS, "code_bytes": CODE_BYTES,
            "head": os.path.basename(args.head), "norm_in": enc.norm_in,
            "source": os.path.basename(args.index), "seed": args.seed,
            "image_root": str(image_root)}
    with open(data_dir / "index_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    idx_mb = (data_dir / "index.bin").stat().st_size / 1024**2
    meta_mb = (data_dir / "meta.json").stat().st_size / 1024**2
    print(f"[build] DONE n={n:,} | index.bin {idx_mb:.2f} MB ({CODE_BYTES} B/row) | "
          f"meta.json {meta_mb:.2f} MB | thumbs -> {thumbs_dir}", flush=True)
    print(f"[build] info: {info}", flush=True)


if __name__ == "__main__":
    main()
