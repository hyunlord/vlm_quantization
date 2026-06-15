"""Benchmark Hamming search backends on real GB10 hardware.

Compares 4 ways to compute top-k Hamming search over a packed binary corpus:
  1. numpy uint8  : np.bitwise_count(xor).sum(axis=1)        (current server.py)
  2. numpy uint64 : same, viewing 32 bytes as 4 uint64       (fewer reduction elems)
  3. matmul ±1    : argmax(corpus_pm1 @ q_pm1) == argmin Hamming  (BLAS, RAM-heavy)
  4. faiss        : IndexBinaryFlat  (HW POPCNT + multithread, keeps packed 32B)

Uses REAL hash codes from an existing demo index (tiled up to target N), so the
distance distribution is realistic. Verifies every backend returns the same
top-k set as the numpy baseline.

Env: INDEX (npz with packed_<bit>), BITS (csv, default 64,256,1024),
     NS (csv N values), K, REPEAT
"""
from __future__ import annotations
import os, time
import numpy as np

INDEX = os.environ.get("INDEX", "/tmp/oi_index.npz")
BITS = [int(x) for x in os.environ.get("BITS", "64,256,1024").split(",")]
NS = [int(x) for x in os.environ.get("NS", "100000,500000,1000000").split(",")]
K = int(os.environ.get("K", "12"))
REPEAT = int(os.environ.get("REPEAT", "25"))


def timeit(fn, repeat=REPEAT):
    fn(); fn()  # warmup
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def tile_to(arr, n):
    reps = (n + len(arr) - 1) // len(arr)
    return np.ascontiguousarray(np.tile(arr, (reps, 1))[:n])


def main():
    idx = np.load(INDEX, allow_pickle=True)
    print(f"index: {INDEX}  real items: {len(idx['ids'])}")
    try:
        import faiss
        print(f"faiss {faiss.__version__}  threads={faiss.omp_get_max_threads()}")
    except Exception as e:
        faiss = None
        print(f"faiss unavailable: {e}")
    import torch
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}\n")

    for BIT in BITS:
        key = f"packed_{BIT}"
        if key not in idx:
            print(f"-- {BIT}-bit: no {key} in index, skip"); continue
        base = idx[key].astype(np.uint8)            # (real, BIT/8)
        qpacked = base[0].copy()                    # query = first item
        print(f"########## {BIT}-bit ({BIT//8} B/item) ##########")
        for N in NS:
            packed = tile_to(base, N)               # (N, BIT/8)
            packed64 = np.ascontiguousarray(packed.view(np.uint64))
            q64 = qpacked.view(np.uint64)
            take = K + K

            # --- baseline top-k (numpy uint8) for correctness check ---
            d8 = np.bitwise_count(np.bitwise_xor(packed, qpacked)).sum(axis=1)
            base_top = set(np.argpartition(d8, take - 1)[:take].tolist())

            res = {}
            res["numpy uint8 (current)"] = (timeit(
                lambda: np.bitwise_count(np.bitwise_xor(packed, qpacked)).sum(axis=1)), 100.0)
            res["numpy uint64"] = (timeit(
                lambda: np.bitwise_count(np.bitwise_xor(packed64, q64)).sum(axis=1)),
                _overlap(np.bitwise_count(np.bitwise_xor(packed64, q64)).sum(axis=1), base_top, take))

            # matmul ±1 (unpack bits -> fp32). RAM heavy but BLAS-fast.
            bits01 = np.unpackbits(packed, axis=1).astype(np.float32)
            pm1 = bits01 * 2 - 1
            qpm1 = (np.unpackbits(qpacked).astype(np.float32) * 2 - 1)
            pm1_t = torch.from_numpy(pm1); qpm1_t = torch.from_numpy(qpm1)
            res["matmul ±1 torch(BLAS)"] = (timeit(lambda: pm1_t @ qpm1_t),
                _overlap(-(pm1 @ qpm1), base_top, take))

            if faiss is not None:
                index = faiss.IndexBinaryFlat(BIT)
                index.add(packed)
                qq = qpacked.reshape(1, -1)
                def f_search():
                    return index.search(qq, take)
                _, fids = index.search(qq, take)
                fset = set(fids[0].tolist())
                res["faiss IndexBinaryFlat"] = (timeit(f_search),
                    len(fset & base_top) / len(base_top) * 100)

            print(f"  N={N:>9,}")
            for name, (ms, ov) in res.items():
                print(f"     {name:26s} {ms:9.3f} ms   top-{take} match {ov:5.0f}%")
            fastest = min(res.items(), key=lambda kv: kv[1][0])
            print(f"     -> fastest: {fastest[0]} ({res['numpy uint8 (current)'][0]/fastest[1][0]:.1f}x vs current)")
        print()


def _overlap(dist, base_top, take):
    top = set(np.argpartition(dist, take - 1)[:take].tolist())
    return len(top & base_top) / len(base_top) * 100


if __name__ == "__main__":
    main()
