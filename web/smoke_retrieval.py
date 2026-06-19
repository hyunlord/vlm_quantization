"""Serving-artifacts retrieval smoke (cycle 2) — EN caption -> image R@1/5/10 over the REAL
web/static/data/index.bin (50K gallery), using the demo's text encoding path end-to-end.

This is a HEALTH CHECK ("not collapsed"), NOT an absolute-threshold benchmark: a 50K
gallery is much harder than the standard COCO 1K/5K, and COCO captions are generic (many
near-duplicate images), so absolute recall is expected to be lower than eval_korean's
5K-gallery numbers. We only confirm caption_i tends to retrieve its own image_i.

Encodes each caption with web.common.Encoder.text_code_packed (the demo's query path; §6.5
already proved server == this offline path), then Hamming-searches index.bin.

Run on DGX (after build_index.py):
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/smoke_retrieval.py [--n 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BITS, CODE_BYTES, Encoder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="EN caption->image retrieval smoke over index.bin")
    p.add_argument("--static", default="web/static")
    p.add_argument("--n", type=int, default=1000, help="sampled query captions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", default="1,5,10")
    args = p.parse_args()

    data = Path(args.static) / "data"
    packed = np.fromfile(data / "index.bin", dtype=np.uint8)
    packed = np.ascontiguousarray(packed.reshape(-1, CODE_BYTES))
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    n_gallery = packed.shape[0]
    if len(meta) != n_gallery:
        raise SystemExit(f"ALIGNMENT BROKEN: meta {len(meta)} != index rows {n_gallery}")

    ks = [int(x) for x in args.ks.split(",")]
    maxk = max(ks)

    # sample gallery rows that have a non-empty caption (sample holds actual row indices)
    have = [i for i, m in enumerate(meta) if (m.get("caption") or "").strip()]
    rng = np.random.default_rng(args.seed)
    if args.n <= 0 or args.n >= len(have):
        sample = have
    else:
        sample = sorted(have[j] for j in rng.choice(len(have), args.n, replace=False).tolist())
    print(f"[smoke] gallery {n_gallery:,} | querying {len(sample)} EN captions "
          f"(seed {args.seed}) | faiss oracle", flush=True)

    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    fidx = faiss.IndexBinaryFlat(CODE_BITS)
    fidx.add(packed)

    enc = Encoder()
    hits = {k: 0 for k in ks}
    t0 = time.perf_counter()
    for c, row in enumerate(sample):
        code = enc.text_code_packed(meta[row]["caption"])
        _, ids = fidx.search(code[None, :], maxk)
        ranked = ids[0].tolist()
        for k in ks:
            if row in ranked[:k]:
                hits[k] += 1
        if (c + 1) % 200 == 0:
            print(f"  {c+1}/{len(sample)} ({(c+1)/(time.perf_counter()-t0):.1f} q/s)", flush=True)

    rec = {f"R@{k}": round(100 * hits[k] / len(sample), 2) for k in ks}
    print(f"\n[smoke] EN caption->image over {n_gallery:,} gallery, "
          f"{len(sample)} queries: " +
          " ".join(f"R@{k} {rec[f'R@{k}']}%" for k in ks), flush=True)
    print("[smoke] note: health check (50K gallery, generic COCO caps) — not an absolute "
          "benchmark; collapse would show near-0.", flush=True)
    print("[smoke] RESULT_JSON " + json.dumps({"gallery": n_gallery, "queries": len(sample),
                                               "recall": rec}), flush=True)


if __name__ == "__main__":
    main()
