"""AAAI rigor §1 — mAP@bit table + FAISS binary-index baseline (eval-only, frozen).

Uses the DEPLOYED ft113 heads (img_h + txt_h, SigLIP2-So400m server path) on the COCO 5K
test set (emb_cache). Nested Matryoshka codes → prefix-slice to each bit. Reports, per bit
and per direction (I→T, T→I): R@{1,5,10} + mAP (instance; single paired relevant per query,
mAP = mean 1/rank). Then a FAISS binary-index micro-benchmark at 1024-bit:
IndexBinaryFlat (exact) vs IndexBinaryIVF (approx) vs numpy brute-force Hamming —
recall@10 (vs exact), per-query latency, index memory — to defend the client-side Hamming choice.

Run on DGX:
  .venv/bin/python web/rigor_map_faiss.py --out_map paper/rigor_map_bit.csv --out_faiss paper/rigor_faiss_bench.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import torch  # noqa
import torch.nn.functional as F  # noqa
from src.models.nested_hash_layer import NestedHashLayer  # noqa

HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
TEST_CACHE = os.environ.get("TEST_CACHE", "/tmp/emb_cache.pt")
BITS = [8, 16, 32, 64, 128, 256, 512, 1024]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def head_z1024(head, x):
    raw = head.hash_head(x)
    length, bn = head.bit_list[-1], head.batch_norms[-1]
    z = F.normalize(bn(raw[:, :length]), p=2, dim=1)
    return z  # (N,1024) pre-sign


def pack(codes_bits01):
    return np.ascontiguousarray(np.packbits(codes_bits01, axis=1, bitorder="big"), dtype=np.uint8)


def load_heads():
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    img_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval()
    txt_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval()
    img_h.load_state_dict(ck["img_h"]); txt_h.load_state_dict(ck["txt_h"])
    for p in list(img_h.parameters()) + list(txt_h.parameters()):
        p.requires_grad_(False)
    return img_h, txt_h


def metrics_binary(qbits, gbits, K=(1, 5, 10)):
    """qbits,gbits: (N,B) bool, paired by row index. Returns dict R@k + mAP (1 relevant)."""
    import faiss
    B = qbits.shape[1]
    ix = faiss.IndexBinaryFlat(B)
    ix.add(pack(gbits.astype(np.uint8)))
    N = qbits.shape[0]
    topk = max(K)
    _, I = ix.search(pack(qbits.astype(np.uint8)), topk)
    out = {}
    for k in K:
        out[f"R{k}"] = round(100 * np.mean([i in I[r, :k] for r, i in enumerate(range(N))]), 2)
    # mAP (single relevant = paired index): need full rank → search all
    _, Ifull = ix.search(pack(qbits.astype(np.uint8)), N)
    rr = []
    for r in range(N):
        pos = np.where(Ifull[r] == r)[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    out["mAP"] = round(100 * float(np.mean(rr)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_map", default="paper/rigor_map_bit.csv")
    ap.add_argument("--out_faiss", default="paper/rigor_faiss_bench.csv")
    args = ap.parse_args()
    t0 = time.perf_counter()

    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    img = F.normalize(EC["img"].float(), dim=1).to(dev)
    txt = F.normalize(EC["txt"].float(), dim=1).to(dev)
    N = img.shape[0]
    img_h, txt_h = load_heads()
    with torch.no_grad():
        zi = (head_z1024(img_h, img) > 0).cpu().numpy()   # (N,1024) bool
        zt = (head_z1024(txt_h, txt) > 0).cpu().numpy()
    print(f"[setup] N={N} bits={BITS} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)

    # ---- mAP@bit table ----
    rows = []
    for b in BITS:
        gi, gt = zi[:, :b], zt[:, :b]
        for direction, q, g in (("I2T", gi, gt), ("T2I", gt, gi)):
            m = metrics_binary(q, g)
            rows.append({"bit": b, "direction": direction, **m})
            print(f"[map] bit{b} {direction}: R@1 {m['R1']} R@5 {m['R5']} R@10 {m['R10']} mAP {m['mAP']}", flush=True)
    with open(args.out_map, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bit", "direction", "R1", "R5", "R10", "mAP"])
        w.writeheader(); w.writerows(rows)
    print(f"[map] wrote {len(rows)} rows -> {args.out_map}", flush=True)

    # ---- FAISS bench @1024 (I2T): exact Flat vs IVF vs numpy brute Hamming ----
    import faiss
    B = 1024
    gpk = pack(zt.astype(np.uint8)); qpk = pack(zi.astype(np.uint8))
    frows = []

    def bench_search(name, searcher, exact_I=None):
        # warmup
        searcher(qpk[:50])
        t = time.perf_counter()
        I = searcher(qpk)
        ms = 1000 * (time.perf_counter() - t) / N
        r10 = round(100 * np.mean([r in I[r, :10] for r in range(N)]), 2)
        if exact_I is not None:
            rec = round(100 * np.mean([len(set(I[r, :10]) & set(exact_I[r, :10])) / 10 for r in range(N)]), 2)
        else:
            rec = 100.0
        return I, {"method": name, "R10": r10, "recall@10_vs_exact": rec,
                   "latency_ms_per_query": round(ms, 4), "index_mem_MB": round(gpk.nbytes / 1e6, 3)}

    fl = faiss.IndexBinaryFlat(B); fl.add(gpk)
    exactI, fr = bench_search("IndexBinaryFlat(exact)", lambda q: fl.search(q, 10)[1]); frows.append(fr)

    # IVF approx
    nlist = 64
    quant = faiss.IndexBinaryFlat(B)
    ivf = faiss.IndexBinaryIVF(quant, B, nlist); ivf.nprobe = 8
    ivf.train(gpk); ivf.add(gpk)
    _, fr = bench_search("IndexBinaryIVF(nlist64,nprobe8)", lambda q: ivf.search(q, 10)[1], exactI); frows.append(fr)

    # numpy brute Hamming (client-side analogue)
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

    def np_hamming(q):
        out = np.empty((q.shape[0], 10), dtype=np.int64)
        for r in range(q.shape[0]):
            d = _LUT[np.bitwise_xor(q[r][None, :], gpk)].sum(1)
            out[r] = np.argpartition(d, 10)[:10][np.argsort(d[np.argpartition(d, 10)[:10]])]
        return out
    _, fr = bench_search("numpy_brute_Hamming(client)", np_hamming, exactI); frows.append(fr)

    with open(args.out_faiss, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "R10", "recall@10_vs_exact", "latency_ms_per_query", "index_mem_MB"])
        w.writeheader(); w.writerows(frows)
    for fr in frows:
        print(f"[faiss] {fr}", flush=True)
    print(f"[done] ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
