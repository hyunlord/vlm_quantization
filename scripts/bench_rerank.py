"""Fair float baseline + coarse-to-fine re-ranking — latency AND recall.

Answers two questions on real OI codes (image->image self-retrieval):
  (1) Fair float baseline: raw numpy emb@q  vs  faiss IndexFlatIP (optimized).
  (2) Re-ranking: short code shortlists, then re-rank by float OR by long bit.
      - bit64 -> float   : best quality, but must KEEP float vectors (no storage win)
      - bit64 -> bit1024 : storage stays small (bits only), quality recovered

Recall@K is measured against full exact-float top-K (the gold standard).

Env: INDEX, K (default 10), M (shortlist depth, default 500), NQ (queries, default 30)
"""
from __future__ import annotations
import os, time
import numpy as np
import faiss

INDEX = os.environ.get("INDEX", "/tmp/oi_index.npz")
K = int(os.environ.get("K", "10"))
M = int(os.environ.get("M", "500"))
NQ = int(os.environ.get("NQ", "30"))
REPEAT = int(os.environ.get("REPEAT", "10"))

idx = np.load(INDEX, allow_pickle=True)
emb = np.ascontiguousarray(idx["emb"].astype(np.float32))      # (N,1152) L2-normalized
pk = {b: np.ascontiguousarray(idx[f"packed_{b}"].astype(np.uint8)) for b in (64, 256, 1024)}
N, D = emb.shape
print(f"index {INDEX}: N={N:,} D={D} | faiss {faiss.__version__} threads={faiss.omp_get_max_threads()} "
      f"| K={K} shortlist M={M} queries={NQ}\n")

# faiss indexes
f_ip = faiss.IndexFlatIP(D); f_ip.add(emb)                     # optimized exact float
b64 = faiss.IndexBinaryFlat(64); b64.add(pk[64])
b256 = faiss.IndexBinaryFlat(256); b256.add(pk[256])

rng = np.random.default_rng(0)
QI = rng.choice(N, size=NQ, replace=False)

# ---- gold standard: exact float top-K (numpy, unambiguous) ----
def gold_topk(qi):
    sc = emb @ emb[qi]
    o = np.argpartition(-sc, K - 1)[:K]
    return set(o.tolist())
GOLD = {int(qi): gold_topk(int(qi)) for qi in QI}

# ---- method definitions: each returns top-K indices for query index qi ----
def m_numpy_float(qi):
    sc = emb @ emb[qi]
    o = np.argpartition(-sc, K - 1)[:K]
    return o[np.argsort(-sc[o])]

def m_faiss_float(qi):
    _, ids = f_ip.search(emb[qi:qi+1], K)
    return ids[0]

def m_faiss_bit256(qi):
    _, ids = b256.search(pk[256][qi:qi+1], K)
    return ids[0]

def m_rerank_float(qi):                       # bit64 shortlist -> float re-rank
    _, cand = b64.search(pk[64][qi:qi+1], M)
    cand = cand[0]
    sc = emb[cand] @ emb[qi]
    top = np.argpartition(-sc, K - 1)[:K]
    return cand[top[np.argsort(-sc[top])]]

def m_rerank_bit1024(qi):                     # bit64 shortlist -> bit1024 re-rank (storage-light)
    _, cand = b64.search(pk[64][qi:qi+1], M)
    cand = cand[0]
    sub = pk[1024][cand].view(np.uint64)
    q = pk[1024][qi].view(np.uint64)
    d = np.bitwise_count(np.bitwise_xor(sub, q)).sum(axis=1)
    top = np.argpartition(d, K - 1)[:K]
    return cand[top[np.argsort(d[top])]]

METHODS = [
    ("numpy float (raw emb@q)", m_numpy_float, D * 4),
    ("faiss IndexFlatIP (float)", m_faiss_float, D * 4),
    ("faiss bit-256", m_faiss_bit256, 32),
    (f"rerank bit64->float (M={M})", m_rerank_float, D * 4 + 8),
    (f"rerank bit64->bit1024 (M={M})", m_rerank_bit1024, 128 + 8),
]

print(f"{'method':32} | {'latency':>9} | {'recall@'+str(K):>9} | {'bytes/img':>9} | note")
print("-" * 92)
for name, fn, bpi in METHODS:
    # latency: median single-query over repeats, averaged across queries
    lat = []
    for qi in QI:
        qi = int(qi)
        fn(qi);
        ts = []
        for _ in range(REPEAT):
            t0 = time.perf_counter(); fn(qi); ts.append(time.perf_counter() - t0)
        lat.append(np.median(ts))
    ms = float(np.median(lat) * 1e3)
    # recall vs gold
    rec = np.mean([len(set(int(x) for x in fn(int(qi))) & GOLD[int(qi)]) / K for qi in QI])
    storage = "float 보관(저장이점X)" if bpi >= D * 4 else "비트만(저장이점O)"
    print(f"{name:32} | {ms:7.3f}ms | {rec*100:7.1f}% | {bpi:>7}B | {storage}")
