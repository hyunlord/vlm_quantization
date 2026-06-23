"""AAAI rigor §1 — classic hashing baselines on the SAME frozen SigLIP2 features.

LSH (random hyperplane) and ITQ (PCA + iterative quantization) computed on frozen SigLIP2-So400m
features, cross-modal I↔T (a SINGLE shared projection across modalities since SigLIP2 already aligns
img/txt in one space). Eval = COCO 5K test, R@{1,5,10}+mAP at bits {16,64,256,1024}, vs our ft113 head
(numbers from rigor_map_bit.csv). Apples-to-apples: identical frozen features, identical eval.

Run on DGX:
  .venv/bin/python web/rigor_baselines.py --out paper/rigor_baselines.csv
"""
from __future__ import annotations
import argparse, csv, os, time
import numpy as np
import torch
import torch.nn.functional as F

TRAIN_CACHE = os.environ.get("TRAIN_CACHE", "/tmp/emb_aug.pt")
TEST_CACHE = os.environ.get("TEST_CACHE", "/tmp/emb_cache.pt")
BITS = [16, 64, 256, 1024]
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pack(b01):
    return np.ascontiguousarray(np.packbits(b01.astype(np.uint8), axis=1, bitorder="big"), dtype=np.uint8)


def metrics(qb, gb, K=(1, 5, 10)):
    import faiss
    B = qb.shape[1]
    ix = faiss.IndexBinaryFlat(B); ix.add(pack(gb))
    N = qb.shape[0]
    _, I = ix.search(pack(qb), max(K))
    out = {f"R{k}": round(100 * np.mean([r in I[r, :k] for r in range(N)]), 2) for k in K}
    _, If = ix.search(pack(qb), N)
    out["mAP"] = round(100 * float(np.mean([1.0 / (np.where(If[r] == r)[0][0] + 1)
                       if len(np.where(If[r] == r)[0]) else 0.0 for r in range(N)])), 2)
    return out


def lsh_codes(feat, W):
    return (feat @ W) > 0  # (N,bits) bool


def itq_fit(X, nbits, n_iter=50):
    """X: (M,D) centered. Returns (mean, PCA components Wp (D,nbits), rotation R (nbits,nbits))."""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Wp = Vt[:nbits].T  # (D,nbits)
    V = Xc @ Wp        # (M,nbits)
    # ITQ rotation
    R = np.linalg.qr(np.random.randn(nbits, nbits))[0]
    for _ in range(n_iter):
        Z = V @ R
        B = np.sign(Z); B[B == 0] = 1
        Up, _, Vtp = np.linalg.svd(B.T @ V)
        R = (Vtp.T @ Up.T)
    return mu, Wp, R


def itq_codes(feat, mu, Wp, R):
    return ((feat - mu) @ Wp @ R) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/rigor_baselines.csv")
    args = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.RandomState(42)

    tr = torch.load(TRAIN_CACHE, map_location="cpu")["train"]
    tr_img = F.normalize(tr["clean"].float(), dim=1).numpy()
    tr_txt = F.normalize(tr["txt"].float(), dim=1).numpy()
    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    te_img = F.normalize(EC["img"].float(), dim=1).numpy()
    te_txt = F.normalize(EC["txt"].float(), dim=1).numpy()
    D = te_img.shape[1]
    # shared fit pool (both modalities, subsample train for ITQ speed)
    pool = np.concatenate([tr_img[:20000], tr_txt[:20000]], 0)
    print(f"[setup] D={D} test={te_img.shape[0]} pool={pool.shape[0]} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)

    rows = []
    for b in BITS:
        # LSH: shared random hyperplanes
        W = rng.randn(D, b).astype(np.float32)
        li, lt = lsh_codes(te_img, W), lsh_codes(te_txt, W)
        # ITQ: shared PCA+rotation fit on pooled train
        mu, Wp, R = itq_fit(pool.astype(np.float64), b)
        ii, it = itq_codes(te_img.astype(np.float64), mu, Wp, R), itq_codes(te_txt.astype(np.float64), mu, Wp, R)
        for name, qi, gt in (("LSH", li, lt), ("ITQ", ii, it)):
            for direction, q, g in (("I2T", qi, gt), ("T2I", gt, qi)):
                m = metrics(q, g)
                rows.append({"method": name, "bit": b, "direction": direction, **m})
                print(f"[bl] {name} bit{b} {direction}: R@1 {m['R1']} R@10 {m['R10']} mAP {m['mAP']}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "bit", "direction", "R1", "R5", "R10", "mAP"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] wrote {len(rows)} rows -> {args.out} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
