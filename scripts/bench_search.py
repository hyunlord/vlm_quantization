"""Benchmark: binary hash search vs float-embedding ('normal') image search.

Reports the two things that matter for this model as a product:

    1. QUALITY  — Recall@k of hash search against the exact float-cosine ranking
                  (i.e. how well the compact codes preserve 'normal' search results).
    2. SPEED/SIZE — index size, query latency and throughput (QPS) of Hamming search
                  vs float cosine, plus the compression ratio.

Runs out-of-the-box on CPU with synthetic data (no trained model needed):

    python scripts/bench_search.py --synthetic --n 100000 --queries 1000 --bit 64

Or against a real index built with --save-emb:

    python scripts/bench_search.py --index indexes/serving.npz --bit 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serve.binary_index import FloatIndex, HammingIndex, has_faiss


def _synthetic(n: int, emb_dim: int, bit: int, queries: int, seed: int, clusters: int = 500):
    """Clustered embeddings (so cosine neighbours are meaningful) + a SimHash code.

    Real image embeddings have cluster structure; pure-random Gaussian data is the
    worst case for *any* hashing method and gives misleadingly low recall. We sample
    cluster centers and add noise, then derive a sign-of-random-projection (SimHash)
    code — a lossy angular code of the embedding, mirroring the trained model.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, emb_dim)).astype(np.float32)
    assign = rng.integers(0, clusters, size=n)
    emb = centers[assign] + 0.15 * rng.standard_normal((n, emb_dim)).astype(np.float32)
    proj = rng.standard_normal((emb_dim, bit)).astype(np.float32)
    codes = np.sign(emb @ proj).astype(np.int8)
    codes[codes == 0] = 1
    q_idx = rng.choice(n, size=queries, replace=False)
    return emb, codes, q_idx, assign


def _recall_at_k(approx_ids: np.ndarray, truth_ids: np.ndarray, k: int) -> float:
    """Mean fraction of the exact top-k recovered by the approximate ranking."""
    hits = 0
    for a, t in zip(approx_ids, truth_ids):
        hits += len(set(a[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(approx_ids) * k)


def _time_search(index, queries, k: int, repeats: int):
    """Return (mean_ms_per_query, qps, ids) timing a batched search `repeats` times."""
    ids, _ = index.search(queries, k)  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        index.search(queries, k)
        best = min(best, time.perf_counter() - t0)
    per_q_ms = best / len(queries) * 1e3
    return per_q_ms, len(queries) / best, ids


def _relevance_at_k(approx_ids, labels, query_labels, k):
    """Fraction of returned top-k that share the query's (cluster) label.

    A semantic 'did we retrieve the right neighbourhood' signal, robust to ties
    among near-identical items (unlike exact float-top-k overlap).
    """
    hits = 0
    for a, ql in zip(approx_ids, query_labels):
        hits += int((labels[a[:k]] == ql).sum())
    return hits / (len(approx_ids) * k)


def main() -> None:
    p = argparse.ArgumentParser(description="Hash vs float image-search benchmark")
    p.add_argument("--index", default=None, help="serving .npz (needs --save-emb)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--emb-dim", type=int, default=1152)
    p.add_argument("--bit", type=int, default=64)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    labels = None
    if args.index:
        from src.serve.binary_index import ServingIndex

        si = ServingIndex.load(args.index)
        if si.emb is None:
            raise SystemExit("index has no 'emb' — rebuild with --save-emb for the baseline")
        bit = args.bit if args.bit in si.per_bit else si.default_bit()
        emb = si.emb
        packed = si.per_bit[bit].packed
        rng = np.random.default_rng(args.seed)
        q_idx = rng.choice(len(emb), size=min(args.queries, len(emb)), replace=False)
        hidx = HammingIndex(packed, n_bits=bit)
        q_codes = np.unpackbits(packed[q_idx], axis=1)[:, :bit] * 2 - 1
        n = len(emb)
    else:
        args.synthetic = True
        emb, codes, q_idx, labels = _synthetic(
            args.n, args.emb_dim, args.bit, args.queries, args.seed
        )
        bit, n = args.bit, args.n
        hidx = HammingIndex.from_codes(codes)
        q_codes = codes[q_idx]

    fidx = FloatIndex(emb)
    q_emb = emb[q_idx]
    q_labels = labels[q_idx] if labels is not None else None

    def _rel(ids):
        return _relevance_at_k(ids, labels, q_labels, args.k) if labels is not None else None

    # Ground truth = exact float cosine ('normal image search')
    f_ms, f_qps, truth_ids = _time_search(fidx, q_emb, args.k, args.repeats)
    h_ms, h_qps, hash_ids = _time_search(hidx, q_codes, args.k, args.repeats)

    rows = [
        ("Float cosine (exact)", fidx.nbytes, f_ms, f_qps, 1.0, _rel(truth_ids)),
        ("Hash Hamming (numpy)", hidx.nbytes, h_ms, h_qps,
         _recall_at_k(hash_ids, truth_ids, args.k), _rel(hash_ids)),
    ]
    faiss_ms = None
    if has_faiss():
        hf = HammingIndex(hidx.packed, n_bits=bit, backend="faiss")
        hf_ms, hf_qps, hf_ids = _time_search(hf, q_codes, args.k, args.repeats)
        faiss_ms = hf_ms
        rows.append(("Hash Hamming (faiss)", hf.nbytes, hf_ms, hf_qps,
                     _recall_at_k(hf_ids, truth_ids, args.k), _rel(hf_ids)))

    print(f"\nN={n:,}  queries={len(q_idx):,}  bit={bit}  k={args.k}  "
          f"faiss={'yes' if has_faiss() else 'no'}\n")
    head = (f"{'method':<24}{'index MB':>10}{'ms/query':>11}{'QPS':>12}"
            f"{'Recall@' + str(args.k):>12}{'Rel@' + str(args.k):>10}")
    print(head)
    print("-" * len(head))
    for name, nbytes, ms, qps, recall, rel in rows:
        rel_s = f"{rel:>10.3f}" if rel is not None else f"{'-':>10}"
        print(f"{name:<24}{nbytes / 1024**2:>10.2f}{ms:>11.3f}"
              f"{qps:>12,.0f}{recall:>12.3f}{rel_s}")
    print("-" * len(head))

    comp = fidx.nbytes / hidx.nbytes
    best_hash_ms = faiss_ms if faiss_ms is not None else h_ms
    backend = "FAISS binary" if faiss_ms is not None else "numpy"
    speedup = f_ms / best_hash_ms if best_hash_ms > 0 else float("inf")
    print(f"size:  {comp:.0f}x smaller index   |   "
          f"speed: {speedup:.0f}x faster per query ({backend}) vs float cosine")
    if faiss_ms is None:
        print("note: install faiss-cpu (.[serve]) for the SIMD popcount backend — "
              "the numpy fallback is not the speed story.")
    print("note: Recall@k = exact float-top-k overlap (strict; untrained-SimHash floor). "
          "Rel@k = fraction sharing the query cluster (semantic quality).")
    print("      the trained model's real retrieval quality is measured by eval.py "
          "(mAP / Recall@K on real data).")


if __name__ == "__main__":
    main()
