"""Three-stage search pipelines compared — latency (broken down) + recall@K.

  full float            : numpy emb@q                       (exact, gold)
  float->float          : faiss IVFFlat (approx coarse) -> exact float   (ANN, big index)
  bit->bit  (256->1024) : bit256 shortlist M -> bit1024 re-rank          (storage-light)
  bit->float (256->flt) : bit256 shortlist M -> float re-rank on M cands (best quality)

Recall@K vs exact-float gold. i2i self-retrieval, real OI codes.
Env: INDEX, K(10), M(500), NQ(30), REPEAT(15)
"""
from __future__ import annotations
import os, time
import numpy as np, faiss

INDEX = os.environ.get("INDEX", "/tmp/oi_index_167k.npz")
K = int(os.environ.get("K", "10")); M = int(os.environ.get("M", "500"))
NQ = int(os.environ.get("NQ", "30")); REPEAT = int(os.environ.get("REPEAT", "15"))

idx = np.load(INDEX, allow_pickle=True)
emb = np.ascontiguousarray(idx["emb"].astype(np.float32))
pk = {b: np.ascontiguousarray(idx[f"packed_{b}"].astype(np.uint8)) for b in (256, 1024)}
N, D = emb.shape
print(f"N={N:,} D={D} | faiss {faiss.__version__} thr={faiss.omp_get_max_threads()} | K={K} M={M} NQ={NQ}\n")

f_ip = faiss.IndexFlatIP(D); f_ip.add(emb)
nlist = max(64, int(np.sqrt(N)))
quant = faiss.IndexFlatIP(D)
ivf = faiss.IndexIVFFlat(quant, D, nlist, faiss.METRIC_INNER_PRODUCT)
ivf.train(emb); ivf.add(emb); ivf.nprobe = 16
b256 = faiss.IndexBinaryFlat(256); b256.add(pk[256])

rng = np.random.default_rng(0); QI = [int(x) for x in rng.choice(N, NQ, replace=False)]
def gold(qi):
    sc = emb @ emb[qi]; return set(np.argpartition(-sc, K-1)[:K].tolist())
GOLD = {qi: gold(qi) for qi in QI}

def full_float(qi):
    sc = emb @ emb[qi]; o = np.argpartition(-sc, K-1)[:K]; return o[np.argsort(-sc[o])]
def f2f(qi):
    _, ids = ivf.search(emb[qi:qi+1], K); return ids[0]
def b2b(qi):
    _, c = b256.search(pk[256][qi:qi+1], M); c = c[0]
    sub = pk[1024][c].view(np.uint64); q = pk[1024][qi].view(np.uint64)
    d = np.bitwise_count(np.bitwise_xor(sub, q)).sum(axis=1)
    t = np.argpartition(d, K-1)[:K]; return c[t[np.argsort(d[t])]]
def b2f(qi):
    _, c = b256.search(pk[256][qi:qi+1], M); c = c[0]
    sc = emb[c] @ emb[qi]; t = np.argpartition(-sc, K-1)[:K]; return c[t[np.argsort(-sc[t])]]

METH = [("full float (gold)", full_float, "4608B  저장이점X"),
        ("float->float (IVF)", f2f, "4608B+  저장이점X"),
        ("bit->bit (256->1024)", b2b, "160B   저장이점O"),
        ("bit->float (256->flt)", b2f, "4608B+ 저장이점X")]

print(f"{'pipeline':24} | {'latency':>9} | {'recall@'+str(K):>9} | 저장")
print("-"*64)
for name, fn, note in METH:
    lat = []
    for qi in QI:
        fn(qi); ts=[]
        for _ in range(REPEAT):
            t0=time.perf_counter(); fn(qi); ts.append(time.perf_counter()-t0)
        lat.append(np.median(ts))
    ms = float(np.median(lat)*1e3)
    rec = float(np.mean([len({int(x) for x in fn(qi)} & GOLD[qi])/K for qi in QI]))
    print(f"{name:24} | {ms:7.3f}ms | {rec*100:7.1f}% | {note}")
