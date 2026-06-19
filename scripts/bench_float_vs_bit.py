"""Float (traditional) vs binary-hash search — before/after the faiss upgrade.

Three full top-k search paths over the SAME tiled corpus (real OI codes):
  - float cosine     : emb @ q  + argpartition           (traditional baseline)
  - bit numpy (old)  : xor->popcount->sum + argpartition  (pre-upgrade)
  - bit faiss (new)  : IndexBinaryFlat.search             (upgraded)

Env: INDEX, BIT (default 256), NS, K, REPEAT
"""
from __future__ import annotations
import os, time
import numpy as np

INDEX = os.environ.get("INDEX", "/tmp/oi_index.npz")
BIT = int(os.environ.get("BIT", "256"))
NS = [int(x) for x in os.environ.get("NS", "100000,500000,1000000").split(",")]
K = int(os.environ.get("K", "12"))
REPEAT = int(os.environ.get("REPEAT", "20"))


def timeit(fn, repeat=REPEAT):
    fn(); fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def tile(arr, n):
    reps = (n + len(arr) - 1) // len(arr)
    return np.ascontiguousarray(np.tile(arr, (reps, 1))[:n])


idx = np.load(INDEX, allow_pickle=True)
emb0 = idx["emb"].astype(np.float32)
pk0 = idx[f"packed_{BIT}"].astype(np.uint8)
import faiss
print(f"index {INDEX}: {len(idx['ids'])} real | BIT={BIT} | faiss {faiss.__version__} "
      f"threads={faiss.omp_get_max_threads()}\n")
print(f"{'N':>10} | {'float cos':>10} | {'bit numpy':>10} | {'bit faiss':>10} | "
      f"{'faiss vs float':>14} | {'faiss vs numpy':>14}")
print("-" * 86)

for N in NS:
    emb = tile(emb0, N); q = emb[0].copy()
    pk = tile(pk0, N); qc = pk0[0].copy()
    pk64 = np.ascontiguousarray(pk.view(np.uint64)); qc64 = qc.view(np.uint64)
    take = K + K

    def s_float():
        sc = emb @ q
        o = np.argpartition(-sc, take - 1)[:take]
        return o[np.argsort(-sc[o])]

    def s_bitnp():
        d = np.bitwise_count(np.bitwise_xor(pk64, qc64)).sum(axis=1)
        o = np.argpartition(d, take - 1)[:take]
        return o[np.argsort(d[o])]

    ix = faiss.IndexBinaryFlat(BIT); ix.add(pk)
    qq = qc.reshape(1, -1)
    def s_faiss():
        return ix.search(qq, take)

    tf = timeit(s_float); tn = timeit(s_bitnp); tx = timeit(s_faiss)
    print(f"{N:>10,} | {tf:>9.2f}m | {tn:>9.2f}m | {tx:>9.2f}m | "
          f"{tf/tx:>12.1f}x | {tn/tx:>12.1f}x")
