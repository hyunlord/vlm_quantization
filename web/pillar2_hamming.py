"""Pillar2 gate v2 — Hamming-space predictors (post-sign code quantities).

After the quadruple negative (cosine/parity/margin all fail to predict R@K), test whether a quantity
computed DIRECTLY from the binary codes predicts binary R@10 across paths/bits AND explains the paradox
head-adapt (parity 81.6%, R 74.0) > distillation (parity 86.9%, R 70.86).

Paths (reuse §3): ceiling / distillation / head-adapt. Gallery = ceiling image codes (deployed index).
Candidates (per the brief):
  (a) neighbor Jaccard@k: query's Hamming-kNN(gallery) vs the CEILING query's kNN — structure preservation.
  (c) Hamming-margin: d(q, paired_img) − min_{j≠paired} d(q, img_j)  (mean + frac>0).  [near-tautological → flagged]
  (d) direction-alignment: mean[ match(q, paired_img) − mean_j match(q, img_j) ]  (alignment to the gallery target).
  (b) answer-rank percentile — diagnostic only (tautological), reported but not a predictor.
Side-by-side: cosine, parity (text-vs-text-ceiling), binR10. Gate = Spearman across 9 (path,bit) pts + per-query.
Run: .venv/bin/python web/pillar2_hamming.py
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer  # noqa

HEAD_PATH = "/tmp/ft_ko_113.pt"; TEST_CACHE = "/tmp/emb_cache.pt"
DISTILL = "/tmp/distill_e5.pt"; TXTH_E5 = "/tmp/txt_h_e5.pt"; E5_TEST_EN = "/tmp/e5_test_en.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
BITS = [64, 256, 1024]
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def mk(state, embed, hidden, bits):
    h = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval(); h.load_state_dict(state); return h


def codes(head, x):  # (N,1024) bool
    raw = head.hash_head(x)
    z = F.normalize(head.batch_norms[-1](raw[:, :1024]), p=2, dim=1)
    return (z > 0).cpu().numpy()


def pack(b):
    return np.ascontiguousarray(np.packbits(b.astype(np.uint8), axis=1, bitorder="big"))


def knn(qb, gb, k):
    import faiss
    ix = faiss.IndexBinaryFlat(qb.shape[1]); ix.add(pack(gb))
    _, I = ix.search(pack(qb), k); return I


def run_distill(caps):
    from transformers import AutoModel, AutoTokenizer
    ck = torch.load(DISTILL, map_location="cpu"); ml = int(ck["maxlen"])
    m = AutoModel.from_pretrained(ck["student"]); m.load_state_dict(ck["backbone"]); m = m.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ck["student"])
    proj = nn.Linear(ck["proj"]["weight"].shape[1], ck["proj"]["weight"].shape[0]); proj.load_state_dict(ck["proj"]); proj = proj.to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(caps), 256):
            t = tok(caps[s:s+256], padding="max_length", max_length=ml, truncation=True, return_tensors="pt")
            o = m(t["input_ids"].to(dev), t["attention_mask"].to(dev)).last_hidden_state
            msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
            out.append(F.normalize(proj((o*msk).sum(1)/msk.sum(1).clamp(min=1e-9)), dim=1).cpu())
    return torch.cat(out, 0).to(dev)


def hamming_to_all(qrow, gp_unpacked):
    # qrow: (B,) bool; gp_unpacked: (N,B) bool → Hamming dists (N,)
    return (qrow[None, :] != gp_unpacked).sum(1)


def main():
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]; embed, hidden = ck["embed"], ck["hidden"]
    img_h = mk(ck["img_h"], embed, hidden, bits); txt_h = mk(ck["txt_h"], embed, hidden, bits)
    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    img = F.normalize(EC["img"].float(), dim=1).to(dev)
    true_txt = F.normalize(EC["txt"].float(), dim=1).to(dev)
    caps = [str(c) for c in EC["captions"]]
    gal = codes(img_h, img)                 # gallery image codes (N,1024)
    true_code = codes(txt_h, true_txt)
    pred = run_distill(caps); pred_code = codes(txt_h, pred)
    e5 = F.normalize(torch.load(E5_TEST_EN, map_location="cpu").float().to(dev), dim=1)
    st = torch.load(TXTH_E5, map_location="cpu"); st = st.get("txt_h", st) if isinstance(st, dict) and "txt_h" in st else st
    ha_code = codes(mk(st, e5.shape[1], hidden, bits), e5)
    cos = {"ceiling": 1.0, "distillation": float(F.cosine_similarity(pred, true_txt, dim=1).mean()), "head-adapt": float("nan")}
    Q = {"ceiling": true_code, "distillation": pred_code, "head-adapt": ha_code}
    N = gal.shape[0]
    print(f"[setup] N={N}", flush=True)

    rows = []; perq = []
    # ceiling kNN per bit for Jaccard reference
    for b in BITS:
        galb = gal[:, :b]
        ceil_knn10 = knn(true_code[:, :b], galb, 10)
        ceil_knn100 = knn(true_code[:, :b], galb, 100)
        galb_bool = galb.astype(bool)
        for name, qc in Q.items():
            qb = qc[:, :b]
            # binR10 + per-query success
            I10 = knn(qb, galb, 10)
            succ = np.array([r in I10[r] for r in range(N)])
            r10 = round(100 * succ.mean(), 2)
            # (a) Jaccard vs ceiling kNN
            qk10 = knn(qb, galb, 10); qk100 = knn(qb, galb, 100)
            jac10 = np.mean([len(set(qk10[r]) & set(ceil_knn10[r])) / len(set(qk10[r]) | set(ceil_knn10[r])) for r in range(N)])
            jac100 = np.mean([len(set(qk100[r]) & set(ceil_knn100[r])) / len(set(qk100[r]) | set(ceil_knn100[r])) for r in range(N)])
            # (c) Hamming-margin + (d) direction-alignment — vectorized via GPU matmul on ±1 codes
            # H = (b - Q·Gᵀ)/2 ; d_true = diag ; d_wrong_min = min over j!=r
            with torch.no_grad():
                Qpm = torch.from_numpy((2 * qb.astype(np.int8) - 1)).float().to(dev)      # (N,b)
                Gpm = torch.from_numpy((2 * galb.astype(np.int8) - 1)).float().to(dev)     # (N,b)
                H = (b - Qpm @ Gpm.t()) / 2.0                                              # (N,N) Hamming
                d_true = H.diag().clone()
                H.fill_diagonal_(float(b + 1))                                             # mask self for wrong-min
                d_wrong_min = H.min(dim=1).values
                d_mean = (H.sum(1) + d_true - (b + 1)) / N    # restore diagonal=d_true into the mean
                hmargin = (d_wrong_min - d_true).cpu().numpy()
                diralign = (((b - d_true) / b) - ((b - d_mean) / b)).cpu().numpy()
            parity = round(100 * float((qc[:, :b] == true_code[:, :b]).mean()), 2)
            rows.append({"path": name, "bit": b, "jaccard@10": round(float(jac10), 4), "jaccard@100": round(float(jac100), 4),
                         "hmargin_mean": round(float(hmargin.mean()), 3), "hmargin_frac_pos": round(float((hmargin > 0).mean()), 4),
                         "diralign_mean": round(float(diralign.mean()), 5),
                         "cosine_to_true": round(cos[name], 4), "parity_pct": parity, "binR10": r10})
            print(f"[h] {name:>12} bit{b}: jac10 {round(float(jac10),3)} hmargin {round(float(hmargin.mean()),2)}(pos {round(float((hmargin>0).mean()),3)}) "
                  f"diralign {round(float(diralign.mean()),4)} | cos {round(cos[name],3)} parity {parity} R10 {r10}", flush=True)
            if b == 1024:
                for r in range(N):
                    perq.append({"path": name, "success": int(succ[r]), "hmargin": int(hmargin[r]), "diralign": round(float(diralign[r]), 5)})

    with open("paper/pillar2_hamming_predictors.csv", "w", newline="") as f:
        cols = ["path", "bit", "jaccard@10", "jaccard@100", "hmargin_mean", "hmargin_frac_pos", "diralign_mean", "cosine_to_true", "parity_pct", "binR10"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open("paper/pillar2_hamming_perquery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "success", "hmargin", "diralign"]); w.writeheader(); w.writerows(perq)

    # gate: Spearman across 9 pts
    from scipy.stats import spearmanr, pointbiserialr
    A = {k: np.array([r[k] for r in rows], float) for k in ("jaccard@10", "jaccard@100", "hmargin_mean", "hmargin_frac_pos", "diralign_mean", "parity_pct", "binR10")}
    print("\n[GATE Spearman vs binR10 across 9 (path,bit) pts]", flush=True)
    for k in ("jaccard@10", "jaccard@100", "hmargin_mean", "hmargin_frac_pos", "diralign_mean", "parity_pct"):
        print(f"  {k:>16}: ρ = {spearmanr(A[k], A['binR10']).correlation:.3f}", flush=True)
    # head-adapt vs distillation @1024 direction
    r = {x["path"]: x for x in rows if x["bit"] == 1024}
    print(f"\n[paradox @1024] head-adapt R10 {r['head-adapt']['binR10']} > distillation {r['distillation']['binR10']} — which predictors agree?", flush=True)
    for k in ("jaccard@10", "hmargin_mean", "diralign_mean", "parity_pct"):
        ha, di = r["head-adapt"][k], r["distillation"][k]
        print(f"  {k:>16}: head-adapt {ha} vs distill {di} → {'CORRECT (ha>di)' if ha > di else 'WRONG (ha<=di)'}", flush=True)
    # per-query @1024
    pa = np.array([p["success"] for p in perq], float);
    for k in ("hmargin", "diralign"):
        v = np.array([p[k] for p in perq], float)
        print(f"[perquery@1024] pointbiserial(success, {k}) = {pointbiserialr(pa, v).correlation:.3f}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
