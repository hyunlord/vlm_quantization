"""D: index scaling latency (paper figure F-scale). Synthetic random 128B codes, single-query
Hamming top-10 latency at N = 50K / 500K / 5M, for faiss IndexBinaryFlat (reference) AND the
actual JS search.js (Node, browser-representative). Warm, median of >=30 runs. EVAL ONLY.

Writes paper/scaling.csv (N, faiss_ms, js_ms, index_MB). Machine: printed (DGX GB10, CPU).
Latency is content-independent (full linear scan), so synthetic random codes are fine.

Run on DGX:  .venv/bin/python web/eval_scaling.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper"
NS = [50_000, 500_000, 5_000_000]
CB, BITS, REPEAT = 128, 1024, 30


def main():
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    PAPER.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    machine = f"DGX GB10 (aarch64), faiss threads={faiss.omp_get_max_threads()}, REPEAT={REPEAT}"
    print(f"[scale] machine: {machine}", flush=True)

    rows = []
    for N in NS:
        db = rng.integers(0, 256, size=(N, CB), dtype=np.uint8)
        ix = faiss.IndexBinaryFlat(BITS); ix.add(db)
        q = np.ascontiguousarray(db[rng.integers(0, N)].reshape(1, -1))
        ix.search(q, 10)  # warm
        ts = []
        for _ in range(REPEAT):
            t0 = time.perf_counter(); ix.search(q, 10); ts.append((time.perf_counter() - t0) * 1e3)
        fm = round(statistics.median(ts), 3)
        rows.append({"N": N, "faiss_ms": fm, "index_MB": round(N * CB / 1e6, 1)})
        print(f"[scale] N={N:,}: faiss median {fm} ms", flush=True)
        del db, ix

    # JS via Node using the ACTUAL search.js (copied to .mjs so Node treats it as ESM).
    tmp_mjs = "/tmp/search_paper.mjs"
    shutil.copy(HERE / "static" / "search.js", tmp_mjs)
    js = {}
    try:
        env = {**os.environ, "SEARCH": tmp_mjs}
        out = subprocess.check_output(["node", str(HERE / "eval_scaling.mjs"), ",".join(map(str, NS))],
                                      env=env, text=True)
        js = json.loads(out.strip().splitlines()[-1])
        print(f"[scale] JS medians: {js}", flush=True)
    except Exception as e:
        print(f"[scale] JS step failed: {e}", flush=True)

    for r in rows:
        r["js_ms"] = js.get(str(r["N"]), "")
    with open(PAPER / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "faiss_ms", "js_ms", "index_MB"]); w.writeheader(); w.writerows(rows)
    print(f"[scale] machine={machine}", flush=True)
    print("[scale] RESULT_JSON " + json.dumps({"rows": rows}), flush=True)
    print("[scale] DONE -> paper/scaling.csv", flush=True)


if __name__ == "__main__":
    main()
