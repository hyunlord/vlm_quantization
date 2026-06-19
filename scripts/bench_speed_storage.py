"""Self-contained search-speed benchmark for the HTML speed/storage sections.
Measures ms/query for top-10 retrieval over a synthetic corpus:
  - float  : L2-normalized cosine (numpy matmul, CPU)
  - bit-np : Hamming via uint8 popcount LUT (numpy, CPU)
  - bit-faiss : faiss IndexBinaryFlat (HW popcount, CPU)
across bit-lengths x corpus sizes. Storage is analytical (computed in HTML).
Random data — measures the SEARCH cost (distance+topk), independent of quality.
Out: /tmp/speed_storage.json
"""
import json, os, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import faiss; HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

D = 1152; Q = 30; K = 10
SIZES = [100_000, 1_000_000]
BITS = [256, 512, 1024]
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
rng = np.random.default_rng(0)


def bench(N, bits):
    out = {}
    # ---- float cosine ----
    fc = rng.standard_normal((N, D)).astype(np.float32); fc /= (np.linalg.norm(fc, axis=1, keepdims=True) + 1e-9)
    fq = rng.standard_normal((Q, D)).astype(np.float32); fq /= (np.linalg.norm(fq, axis=1, keepdims=True) + 1e-9)
    t = time.perf_counter()
    for q in fq:
        s = fc @ q; np.argpartition(-s, K)[:K]
    out["float_ms"] = round((time.perf_counter() - t) / Q * 1000, 3)
    del fc
    # ---- bit ----
    B = bits // 8
    bc = rng.integers(0, 256, (N, B), dtype=np.uint8)
    bq = rng.integers(0, 256, (Q, B), dtype=np.uint8)
    t = time.perf_counter()
    for q in bq:
        d = LUT[np.bitwise_xor(bc, q)].sum(1); np.argpartition(d, K)[:K]
    out["bit_np_ms"] = round((time.perf_counter() - t) / Q * 1000, 3)
    if HAVE_FAISS:
        idx = faiss.IndexBinaryFlat(bits); idx.add(bc)
        t = time.perf_counter()
        for q in bq:
            idx.search(q.reshape(1, -1), K)
        out["bit_faiss_ms"] = round((time.perf_counter() - t) / Q * 1000, 3)
    out["speedup_faiss_vs_float"] = round(out["float_ms"] / out["bit_faiss_ms"], 1) if HAVE_FAISS and out.get("bit_faiss_ms") else None
    del bc
    return out


res = {"meta": {"D": D, "Q": Q, "K": K, "faiss": HAVE_FAISS, "device": "GB10 CPU"}, "bench": {}}
for N in SIZES:
    res["bench"][str(N)] = {}
    for bits in BITS:
        r = bench(N, bits)
        res["bench"][str(N)][str(bits)] = r
        print(f"N={N:>9} {bits}b | float {r['float_ms']:8} ms | bit-np {r['bit_np_ms']:8} | "
              f"bit-faiss {r.get('bit_faiss_ms','-'):8} | faiss vs float x{r.get('speedup_faiss_vs_float','-')}", flush=True)
json.dump(res, open("/tmp/speed_storage.json", "w"), indent=2)
print("SPEED_BENCH_DONE", flush=True)
