"""AAAI rigor §4 — multilingual binary robustness (XM3600, 36 lang), eval-only.

For each language: text→image retrieval over the 3600 XM3600 gallery, comparing
CONTINUOUS SigLIP2-text cosine vs 1024-bit BINARY (deployed ft113 txt_h/img_h codes).
Reports per-lang R@{1,5,10} (continuous, binary) and the binarization loss Δ = cont − bin,
plus bit {256,1024}. Emphasis on non-Latin scripts (ko/th/hi/bn/ar/el/...).

Cache /tmp/xm_so400m_lc.pt: {img_emb (3600,1152), per_lang{lang:{text_emb,gold}}} (lowercased text tower).
Run on DGX:
  .venv/bin/python web/rigor_multiling.py --out paper/rigor_multiling.csv
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
XM = os.environ.get("XM", "/tmp/xm_so400m_lc.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
# rough script class for non-Latin emphasis
NONLATIN = {"ko", "ja", "zh", "th", "hi", "bn", "te", "ar", "fa", "el", "ru", "uk", "he", "ko_KR"}


def head_z1024(head, x):
    raw = head.hash_head(x)
    return F.normalize(head.batch_norms[-1](raw[:, :head.bit_list[-1]]), p=2, dim=1)


def load_heads():
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    img_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval()
    txt_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval()
    img_h.load_state_dict(ck["img_h"]); txt_h.load_state_dict(ck["txt_h"])
    return img_h, txt_h


def recall_cont(q, g, gold, K=(1, 5, 10)):
    # cosine: q,g L2-normed → rank by dot, descending
    sims = q @ g.T  # (Nt, Ng)
    order = np.argsort(-sims, axis=1)
    return {k: round(100 * np.mean([gold[i] in order[i, :k] for i in range(len(gold))]), 2) for k in K}


def recall_bin(qb, gb, gold, bits, K=(1, 5, 10)):
    # qb,gb: (N,1024) bool; rank by Hamming over prefix `bits` (faiss, fast)
    import faiss
    qp = np.ascontiguousarray(np.packbits(qb[:, :bits].astype(np.uint8), axis=1, bitorder="big"))
    gp = np.ascontiguousarray(np.packbits(gb[:, :bits].astype(np.uint8), axis=1, bitorder="big"))
    ix = faiss.IndexBinaryFlat(bits); ix.add(gp)
    _, I = ix.search(qp, max(K))
    return {k: round(100 * np.mean([gold[i] in I[i, :k] for i in range(len(gold))]), 2) for k in K}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/rigor_multiling.csv")
    args = ap.parse_args()
    D = torch.load(XM, map_location="cpu")
    img = F.normalize(D["img_emb"].float(), dim=1)
    img_h, txt_h = load_heads()
    with torch.no_grad():
        img_cont = img.numpy()
        img_bin = (head_z1024(img_h, img.to(dev)) > 0).cpu().numpy()
    rows = []
    for lang, d in D["per_lang"].items():
        txt = F.normalize(d["text_emb"].float(), dim=1)
        gold = np.asarray(d["gold"]).reshape(-1).astype(int)
        with torch.no_grad():
            txt_cont = txt.numpy()
            txt_bin = (head_z1024(txt_h, txt.to(dev)) > 0).cpu().numpy()
        rc = recall_cont(txt_cont, img_cont, gold)
        rb1024 = recall_bin(txt_bin, img_bin, gold, 1024)
        rb256 = recall_bin(txt_bin, img_bin, gold, 256)
        rows.append({
            "lang": lang, "nonlatin": int(lang in NONLATIN), "n": len(gold),
            "cont_R1": rc[1], "cont_R5": rc[5], "cont_R10": rc[10],
            "bin1024_R1": rb1024[1], "bin1024_R5": rb1024[5], "bin1024_R10": rb1024[10],
            "bin256_R10": rb256[10],
            "delta_R10": round(rc[10] - rb1024[10], 2),
        })
        print(f"[xm] {lang:>3} nonLatin={int(lang in NONLATIN)} cont_R10 {rc[10]} bin1024_R10 {rb1024[10]} Δ {round(rc[10]-rb1024[10],2)}", flush=True)
    # summary
    lat = [r for r in rows if not r["nonlatin"]]
    nl = [r for r in rows if r["nonlatin"]]
    print(f"[summary] Latin Δ_R10 mean {np.mean([r['delta_R10'] for r in lat]):.2f} | nonLatin Δ_R10 mean {np.mean([r['delta_R10'] for r in nl]):.2f}", flush=True)
    cols = ["lang", "nonlatin", "n", "cont_R1", "cont_R5", "cont_R10",
            "bin1024_R1", "bin1024_R5", "bin1024_R10", "bin256_R10", "delta_R10"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[done] wrote {len(rows)} langs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
