"""Pillar 2 margin theory — does pre-sign margin |z| predict retrieval where cosine/parity fail?

Three text paths into the frozen ft113 code space (reuse §3): ceiling (true SigLIP-text),
distillation (e5->pred SigLIP-text via distill_e5), head-adapt (e5->txt_h'). Gallery = img_h(img).
pre-sign z = L2norm(BN(hash_head(x)))[:, :B] (continuous, before sign).

Noise σ from §2 (encoder-output int8 quant) — no new assumption. Outputs:
  paper/pillar2_margin_dist.csv       (1) |z| distribution per path×bit
  paper/pillar2_flip_vs_margin.csv    (2) empirical flip vs Φ(−|z|/σ), binned by |z|
  paper/pillar2_margin_predicts_R.csv (3) GATE: D=mean Φ(−|z|/σ) vs binR@10 vs cosine/parity
  paper/pillar2_perquery.csv          (3) per-query D_q vs retrieval success
Run: .venv/bin/python web/pillar2_margin.py
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.special import ndtr  # Φ

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer  # noqa

HEAD_PATH = "/tmp/ft_ko_113.pt"; TEST_CACHE = "/tmp/emb_cache.pt"
DISTILL = "/tmp/distill_e5.pt"; TXTH_E5 = "/tmp/txt_h_e5.pt"; E5_TEST_EN = "/tmp/e5_test_en.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
BITS = [64, 256, 1024]
TAUS = [0.005, 0.01, 0.02, 0.05]


def mk_head(state, embed, hidden, bits):
    h = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval(); h.load_state_dict(state); return h


def z_full(head, x):  # continuous pre-sign (N,1024)
    raw = head.hash_head(x)
    return F.normalize(head.batch_norms[-1](raw[:, :1024]), p=2, dim=1)


def q_int8(x):
    s = x.abs().max() / 127.0
    return torch.round(x / s).clamp(-127, 127) * s


def pack(b01):
    return np.ascontiguousarray(np.packbits(b01.astype(np.uint8), axis=1, bitorder="big"))


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def bin_r10(qz, gz, b):
    import faiss
    qb = (_np(qz)[:, :b] > 0); gb = (_np(gz)[:, :b] > 0)
    ix = faiss.IndexBinaryFlat(b); ix.add(pack(gb))
    _, I = ix.search(pack(qb), 10)
    succ = np.array([r in I[r] for r in range(qb.shape[0])])
    return round(100 * succ.mean(), 2), succ


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


def main():
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]; embed, hidden = ck["embed"], ck["hidden"]
    img_h = mk_head(ck["img_h"], embed, hidden, bits); txt_h = mk_head(ck["txt_h"], embed, hidden, bits)
    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    img = F.normalize(EC["img"].float(), dim=1).to(dev)
    true_txt = F.normalize(EC["txt"].float(), dim=1).to(dev)
    caps = [str(c) for c in EC["captions"]]
    with torch.no_grad():
        gz = z_full(img_h, img)                      # gallery pre-sign
        true_z = z_full(txt_h, true_txt)
        pred = run_distill(caps); pred_z = z_full(txt_h, pred)
        e5 = F.normalize(torch.load(E5_TEST_EN, map_location="cpu").float().to(dev), dim=1)
        st = torch.load(TXTH_E5, map_location="cpu"); st = st.get("txt_h", st) if isinstance(st, dict) and "txt_h" in st else st
        ha = mk_head(st, e5.shape[1], hidden, bits); ha_z = z_full(ha, e5)
    true_code = (true_z > 0).cpu().numpy()
    paths = {  # name: (text pre-sign z, feature, head, cosine-to-true)
        "ceiling":      (true_z, true_txt, txt_h, 1.0),
        "distillation": (pred_z, pred,     txt_h, float(F.cosine_similarity(pred, true_txt, dim=1).mean())),
        "head-adapt":   (ha_z,   e5,       ha,    float("nan")),
    }

    # σ from §2: int8 on the feature, std of induced z perturbation (use ceiling path / true features)
    with torch.no_grad():
        true_z_q = z_full(txt_h, F.normalize(q_int8(true_txt), dim=1))
        gz_q = z_full(img_h, F.normalize(q_int8(img), dim=1))
    sigma = float((true_z_q - true_z).std())
    print(f"[sigma] int8-induced z-noise σ = {sigma:.5f}", flush=True)

    # ---- (1) margin distribution ----
    d1 = []
    for name, (z, feat, head, cos) in paths.items():
        az = z.abs().cpu().numpy()
        for b in BITS:
            azb = az[:, :b].ravel()
            row = {"path": name, "bit": b, "mean_absz": round(float(azb.mean()), 5),
                   "median_absz": round(float(np.median(azb)), 5), "p10_absz": round(float(np.percentile(azb, 10)), 5)}
            for t in TAUS:
                row[f"frac_below_{t}"] = round(float((azb < t).mean()), 4)
            d1.append(row)
    cols1 = ["path", "bit", "mean_absz", "median_absz", "p10_absz"] + [f"frac_below_{t}" for t in TAUS]
    with open("paper/pillar2_margin_dist.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols1); w.writeheader(); w.writerows(d1)

    # ---- (2) flip vs margin (ceiling path, int8 noise) ----
    az = true_z.abs().cpu().numpy().ravel()
    flip = (torch.sign(true_z_q) != torch.sign(true_z)).cpu().numpy().ravel()
    edges = np.linspace(0, np.percentile(az, 99), 31)
    d2 = []
    for i in range(len(edges)-1):
        m = (az >= edges[i]) & (az < edges[i+1])
        if m.sum() < 50: continue
        c = 0.5*(edges[i]+edges[i+1])
        emp = float(flip[m].mean()); pred_g = float(ndtr(-c/sigma))
        d2.append({"absz_bin": round(c, 5), "emp_flip": round(emp, 5), "gauss_pred": round(pred_g, 5), "n": int(m.sum())})
    # R^2 of emp vs gauss
    e = np.array([r["emp_flip"] for r in d2]); g = np.array([r["gauss_pred"] for r in d2]); n = np.array([r["n"] for r in d2])
    ss_res = float((n*(e-g)**2).sum()); ss_tot = float((n*(e-e.mean())**2).sum())
    r2 = round(1 - ss_res/ss_tot, 4) if ss_tot > 0 else float("nan")
    with open("paper/pillar2_flip_vs_margin.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["absz_bin", "emp_flip", "gauss_pred", "n"]); w.writeheader(); w.writerows(d2)
    print(f"[flip] weighted R^2(emp vs Φ(−|z|/σ)) = {r2}", flush=True)

    # ---- (3) GATE: D vs binR@10 vs cosine/parity ----
    d3 = []
    for name, (z, feat, head, cos) in paths.items():
        zc = z.cpu().numpy()
        code = (z > 0).cpu().numpy()
        parity = round(100*float((code == true_code).mean()), 2)
        for b in BITS:
            D = float(ndtr(-np.abs(zc[:, :b])/sigma).mean())  # expected per-bit distortion
            r10, _ = bin_r10(z, gz, b)
            d3.append({"path": name, "bit": b, "D_expected_distort": round(D, 5),
                       "mean_absz": round(float(np.abs(zc[:, :b]).mean()), 5),
                       "frac_below_0.01": round(float((np.abs(zc[:, :b]) < 0.01).mean()), 4),
                       "cosine_to_true": round(cos, 4), "parity_pct": parity, "binR10": r10})
            print(f"[gate] {name:>12} bit{b}: D {round(D,4)} mean|z| {round(float(np.abs(zc[:,:b]).mean()),4)} cos {round(cos,3)} parity {parity} R10 {r10}", flush=True)
    with open("paper/pillar2_margin_predicts_R.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","bit","D_expected_distort","mean_absz","frac_below_0.01","cosine_to_true","parity_pct","binR10"]); w.writeheader(); w.writerows(d3)
    # correlations across the 9 (path,bit) points
    from scipy.stats import spearmanr
    arr = {k: np.array([r[k] for r in d3]) for k in ("D_expected_distort","parity_pct","binR10","mean_absz")}
    sp_D = spearmanr(arr["D_expected_distort"], arr["binR10"]).correlation
    sp_m = spearmanr(arr["mean_absz"], arr["binR10"]).correlation
    sp_p = spearmanr(arr["parity_pct"], arr["binR10"]).correlation
    print(f"[corr across 9 pts] Spearman(D, R10)={sp_D:.3f}  (mean|z|,R10)={sp_m:.3f}  (parity,R10)={sp_p:.3f}", flush=True)

    # ---- per-query: D_q vs success (ceiling, 1024) ----
    _, succ = bin_r10(true_z, gz, 1024)
    zc = true_z.abs().cpu().numpy()
    Dq = ndtr(-zc/sigma).mean(1)  # per-query expected distortion
    nlow = (zc < 0.01).sum(1)
    from scipy.stats import pointbiserialr
    pb = pointbiserialr(succ.astype(float), Dq).correlation
    # success rate by D_q quintile
    q = np.quantile(Dq, [0,.2,.4,.6,.8,1.0])
    dq_rows = []
    for i in range(5):
        m = (Dq >= q[i]) & (Dq <= q[i+1] if i==4 else Dq < q[i+1])
        dq_rows.append({"Dq_quintile": i+1, "Dq_lo": round(float(q[i]),5), "Dq_hi": round(float(q[i+1]),5),
                        "success_rate": round(100*float(succ[m].mean()),2), "n": int(m.sum())})
    with open("paper/pillar2_perquery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Dq_quintile","Dq_lo","Dq_hi","success_rate","n"]); w.writeheader(); w.writerows(dq_rows)
    print(f"[perquery] pointbiserial(success, D_q)={pb:.3f}; success by D_q quintile: {[r['success_rate'] for r in dq_rows]}", flush=True)

    # ---- (4) causal mini-check: margin scale-up (sign-invariant) clean vs noisy ----
    with torch.no_grad():
        clean_r10, _ = bin_r10(true_z, gz, 1024)
        # scale |z| up 5x (sign invariant) then re-add same int8 noise magnitude → fewer flips?
        z_scaled = true_z * 5.0
        noise = (true_z_q - true_z)  # same realized noise
        z_scaled_noisy = z_scaled + noise
        r10_orig_noisy, _ = bin_r10(true_z + noise, gz, 1024)
        r10_scaled_noisy, _ = bin_r10(z_scaled_noisy, gz, 1024)
    print(f"[causal] clean R10 {clean_r10} | orig+noise {r10_orig_noisy} | 5x-margin+noise {r10_scaled_noisy} "
          f"(clean sign-invariant to scaling; gain under noise = margin is causal there)", flush=True)
    print(f"[done] σ={sigma:.5f} flipR2={r2} Spearman(D,R10)={sp_D:.3f} vs (parity,R10)={sp_p:.3f}", flush=True)


if __name__ == "__main__":
    main()
