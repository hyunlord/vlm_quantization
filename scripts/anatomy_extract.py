"""Anatomy extraction (A/B/C) — backbone embeddings -> head stages -> code space.

Exploration, not gating. Dumps raw CSVs (paper/anatomy_*.csv) + sampled arrays
(paper/anatomy_arrays.npz) for plotting. Three text paths into the frozen ft113 anchor:
  ceiling      : true SigLIP-text -> txt_h
  distillation : e5 -> distill(student+proj) -> pred SigLIP-text -> txt_h
  head-adapt   : e5 -> txt_h'  (txt_h_e5.pt)
Image anchor (gallery): frozen SigLIP-img -> img_h. pre-sign z = L2norm(BN(hash_head(x)[:,:1024])).
Neutral extraction only — no interpretation. Run: .venv/bin/python scripts/anatomy_extract.py
"""
from __future__ import annotations
import csv, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
HEAD_PATH = "/tmp/ft_ko_113.pt"; EC_PATH = "/tmp/emb_cache.pt"
DISTILL = "/tmp/distill_e5.pt"; TXTH_E5 = "/tmp/txt_h_e5.pt"; E5_EN = "/tmp/e5_test_en.pt"
OUT = os.path.join(REPO, "paper"); os.makedirs(OUT, exist_ok=True)
SAMPLE = 1500  # rows kept for scatter/hist arrays
torch.manual_seed(0)


def shist(arr, bins, rng=None):
    """histogram that tolerates degenerate (constant) ranges."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0]), np.array([0.0, 1.0])
    lo, hi = (rng if rng is not None else (float(arr.min()), float(arr.max())))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        c = float(arr.mean() if arr.size else 0.0)
        return np.array([arr.size]), np.array([c - 0.5, c + 0.5])
    return np.histogram(arr, bins=bins, range=(lo, hi))


def wcsv(name, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"  wrote paper/{name} ({len(rows)} rows)", flush=True)


def mk_head(state, embed, hidden, bits):
    h = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval(); h.load_state_dict(state); return h


def stages(head, x, B=1024):
    """return sliced (pre-BN), bn (post-BN pre-L2), z (post-L2 pre-sign), code (sign)."""
    raw = head.hash_head(x)[:, :B]
    bn = head.batch_norms[-1](raw)
    z = F.normalize(bn, p=2, dim=1)
    return {"sliced": raw, "bn": bn, "z": z, "code": torch.sign(z)}


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


# ---------------- load ----------------
ck = torch.load(HEAD_PATH, map_location="cpu")
bits, embed, hidden = [int(b) for b in ck["bits"]], ck["embed"], ck["hidden"]
img_h = mk_head(ck["img_h"], embed, hidden, bits); txt_h = mk_head(ck["txt_h"], embed, hidden, bits)
EC = torch.load(EC_PATH, map_location="cpu")["test"]
img_raw = EC["img"].float().to(dev); txt_raw = EC["txt"].float().to(dev)
caps = [str(c) for c in EC["captions"]]
ids = EC["ids"]; labels = (ids if torch.is_tensor(ids) else torch.tensor([int(x) for x in ids])).to(dev)
img = F.normalize(img_raw, dim=1); true_txt = F.normalize(txt_raw, dim=1)
N = img.shape[0]
print(f"N={N} bits={bits} embed={embed}", flush=True)

with torch.no_grad():
    pred_raw = run_distill(caps)                       # distill pred SigLIP-text (L2 already)
    e5_raw = torch.load(E5_EN, map_location="cpu").float().to(dev); e5 = F.normalize(e5_raw, dim=1)
    st = torch.load(TXTH_E5, map_location="cpu"); st = st["txt_h"] if isinstance(st, dict) and "txt_h" in st else st
    ha_head = mk_head(st, e5.shape[1], hidden, bits)
    # backbone embeddings per source (post-L2 used for cosine/stages; raw kept for norms)
    SRC = {  # name: (raw_unnormed, normed, head_for_stages)
        "image":        (img_raw, img, img_h),
        "ceiling":      (txt_raw, true_txt, txt_h),
        "distillation": (pred_raw, pred_raw, txt_h),   # distill output already L2
        "head-adapt":   (e5_raw, e5, ha_head),
    }
    ST = {name: stages(h, x) for name, (_, x, h) in SRC.items()}

# =================== A) backbone embedding space ===================
print("[A] backbone embeddings", flush=True)
sv_rows, var_rows, norm_rows = [], [], []
for name, (raw, normed, _) in SRC.items():
    X = normed - normed.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(X.float()).cpu().numpy()
    sv2 = sv**2; pr = float((sv2.sum()**2) / (sv2**2).sum())   # participation ratio (effective dim)
    for i, s in enumerate(sv[:200]):
        sv_rows.append({"source": name, "rank": i, "singular_value": round(float(s), 6),
                        "var_frac": round(float(sv2[i]/sv2.sum()), 6)})
    pv = normed.var(0).cpu().numpy()
    for i in range(0, normed.shape[1]):
        var_rows.append({"source": name, "dim": i, "var": round(float(pv[i]), 8)})
    nrm = raw.norm(dim=1).cpu().numpy()
    hist, edges = shist(nrm, 60)
    for c, e0 in zip(hist, edges[:-1]):
        norm_rows.append({"source": name, "norm_bin": round(float(e0), 4), "count": int(c)})
    print(f"   {name:>12}: eff_dim(PR)={pr:.1f}/{normed.shape[1]}  rawnorm[min/med/max]={nrm.min():.2f}/{np.median(nrm):.2f}/{nrm.max():.2f}", flush=True)
    sv_rows.append({"source": name, "rank": -1, "singular_value": round(pr, 3), "var_frac": -1})  # PR sentinel
wcsv("anatomy_A_svspectrum.csv", sv_rows)
wcsv("anatomy_A_perdim_var.csv", var_rows)
wcsv("anatomy_A_norm_hist.csv", norm_rows)

# pair cosine vs random (text paths vs image)
cos_rows = []
perm = torch.randperm(N, device=dev)
for name, (_, normed, _) in SRC.items():
    if name == "image" or normed.shape[1] != img.shape[1]:
        continue  # cross-modal cosine only meaningful in shared SigLIP space (skip 384-d e5 head-adapt; its pairing is in code space, see C)
    pair = F.cosine_similarity(normed, img, dim=1).cpu().numpy()           # paired text-image
    rand = F.cosine_similarity(normed, img[perm], dim=1).cpu().numpy()      # random
    for tag, arr in [("pair", pair), ("random", rand)]:
        h, e = shist(arr, 60, (-0.2, 1.0))
        for c, e0 in zip(h, e[:-1]):
            cos_rows.append({"path": name, "kind": tag, "cos_bin": round(float(e0), 4), "count": int(c)})
wcsv("anatomy_A_cosine_pair_vs_random.csv", cos_rows)

# =================== B) head stages ===================
print("[B] head stages", flush=True)
stage_var, stage_hist, absz_rows = [], [], []
for name in SRC:
    for sname in ("sliced", "bn", "z"):
        t = ST[name][sname]
        pm, pv = t.mean(0).cpu().numpy(), t.var(0).cpu().numpy()
        for i in range(0, t.shape[1], 1):
            stage_var.append({"path": name, "stage": sname, "dim": i,
                              "mean": round(float(pm[i]), 6), "var": round(float(pv[i]), 8)})
        flat = t.flatten().cpu().numpy()
        lo, hi = np.percentile(flat, [0.5, 99.5])
        h, e = shist(flat, 80, (lo, hi))
        for c, e0 in zip(h, e[:-1]):
            stage_hist.append({"path": name, "stage": sname, "val_bin": round(float(e0), 6), "count": int(c)})
    az = ST[name]["z"].abs().flatten().cpu().numpy()
    h, e = shist(az, 100, (0, float(np.percentile(az, 99.5))))
    for c, e0 in zip(h, e[:-1]):
        absz_rows.append({"path": name, "absz_bin": round(float(e0), 6), "count": int(c)})
    for thr in (0.001, 0.005, 0.01, 0.02, 0.05):
        absz_rows.append({"path": name, "absz_bin": f"frac_below_{thr}", "count": round(float((az < thr).mean()), 5)})
    print(f"   {name:>12}: mean|z|={az.mean():.5f} frac|z|<0.01={float((az<0.01).mean()):.4f}", flush=True)
wcsv("anatomy_B_stage_perdim.csv", stage_var)
wcsv("anatomy_B_stage_hist.csv", stage_hist)
wcsv("anatomy_B_absz.csv", absz_rows)

# =================== C) code space ===================
print("[C] code space", flush=True)
codes = {name: ST[name]["code"] for name in SRC}          # (N,1024) ±1
gal = codes["image"]
bal_rows, ent_rows = [], []
corr_mats = {}
for name in SRC:
    c = codes[name]
    pos = ((c > 0).float().mean(0)).cpu().numpy()          # per-bit +1 fraction
    p = np.clip(pos, 1e-6, 1-1e-6); H = -(p*np.log2(p)+(1-p)*np.log2(1-p))
    for i in range(1024):
        bal_rows.append({"path": name, "bit": i, "frac_pos": round(float(pos[i]), 5), "entropy": round(float(H[i]), 5)})
    cm = np.corrcoef(c.cpu().numpy().T)                    # 1024x1024 bit-bit corr
    corr_mats[name] = cm.astype(np.float16)
    off = cm[~np.eye(1024, dtype=bool)]
    rank = int(torch.linalg.matrix_rank(c.float()).item())
    ent_rows.append({"path": name, "sum_bit_entropy": round(float(H.sum()), 2),
                     "mean_abs_offcorr": round(float(np.abs(off).mean()), 5),
                     "p99_abs_offcorr": round(float(np.percentile(np.abs(off), 99)), 5),
                     "code_rank": rank})
    print(f"   {name:>12}: sumH={H.sum():.1f}/1024 rank={rank} mean|offcorr|={np.abs(off).mean():.4f}", flush=True)
wcsv("anatomy_C_bit_balance.csv", bal_rows)
wcsv("anatomy_C_code_summary.csv", ent_rows)


def hamming(q, g, b):
    return ((b - q[:, :b] @ g[:, :b].t()) / 2)


# Hamming distributions: pair vs nearest-wrong, per path (at 1024)
ham_rows = []
for name in SRC:
    if name == "image":
        continue
    d = hamming(codes[name], gal, 1024)
    pair = d[torch.arange(N), torch.arange(N)].cpu().numpy()
    dd = d.clone(); dd[torch.arange(N), torch.arange(N)] = 1e9
    nwrong = dd.min(1).values.cpu().numpy()
    for tag, arr in [("pair", pair), ("nearest_wrong", nwrong)]:
        h, e = shist(arr, 80, (0, 1024))
        for cc, e0 in zip(h, e[:-1]):
            ham_rows.append({"path": name, "kind": tag, "ham_bin": int(e0), "count": int(cc)})
    gap = nwrong - pair
    h, e = shist(gap, 80, (-512, 512))
    for cc, e0 in zip(h, e[:-1]):
        ham_rows.append({"path": name, "kind": "gap(nwrong-pair)", "ham_bin": int(e0), "count": int(cc)})
    print(f"   {name:>12}: pairHam[med]={np.median(pair):.0f} nwrong[med]={np.median(nwrong):.0f} gap[med]={np.median(gap):.0f}", flush=True)
wcsv("anatomy_C_hamming_dist.csv", ham_rows)

# bit-ablation (grouped: 16 groups of 64) — R@10 drop when a group is removed, per path
def r10(qc, gc, keep):
    d = ((keep.sum().item() - (qc[:, keep]) @ gc[:, keep].t()) / 2)
    order = d.argsort(1)
    rel = (labels[order] == labels[:, None])
    return round((rel[:, :10].sum(1) > 0).float().mean().item()*100, 2)


abl_rows = []
G = 16; gs = 1024 // G
for name in SRC:
    if name == "image":
        continue
    qc = codes[name]
    full = r10(qc, gal, torch.ones(1024, dtype=torch.bool, device=dev))
    for g in range(G):
        keep = torch.ones(1024, dtype=torch.bool, device=dev); keep[g*gs:(g+1)*gs] = False
        abl_rows.append({"path": name, "group": g, "bits": f"{g*gs}-{(g+1)*gs-1}",
                         "r10_drop": round(full - r10(qc, gal, keep), 3), "full_r10": full})
    print(f"   {name:>12}: full R@10 {full}", flush=True)
wcsv("anatomy_C_bit_ablation_grouped.csv", abl_rows)

# ---------------- sampled arrays for plots ----------------
idx = torch.randperm(N)[:SAMPLE].to(dev)
np.savez_compressed(os.path.join(OUT, "anatomy_arrays.npz"),
                    sample_idx=idx.cpu().numpy(),
                    **{f"emb_{n}": SRC[n][1][idx].cpu().numpy() for n in SRC},
                    **{f"z_{n}": ST[n]["z"][idx].cpu().numpy().astype(np.float16) for n in SRC},
                    **{f"code_{n}": codes[n][idx].cpu().numpy().astype(np.int8) for n in SRC},
                    **{f"corr_{n}": corr_mats[n] for n in SRC})
print("ANATOMY_ABC_DONE -> paper/anatomy_*.csv + anatomy_arrays.npz", flush=True)
