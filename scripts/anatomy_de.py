"""Anatomy extraction D (path comparison) + E (multilingual/on-device).
Reuses ft113 anchor + 3 paths (D, COCO EN test) and XM3600 so400m (E, ceiling path).
Dumps paper/anatomy_D_*.csv, paper/anatomy_E_*.csv + anatomy_de_arrays.npz.
Neutral extraction only. Run: .venv/bin/python scripts/anatomy_de.py
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
HEAD_PATH = "/tmp/ft_ko_113.pt"; EC_PATH = "/tmp/emb_cache.pt"
DISTILL = "/tmp/distill_e5.pt"; TXTH_E5 = "/tmp/txt_h_e5.pt"; E5_EN = "/tmp/e5_test_en.pt"
XM = "/tmp/xm_so400m_lc.pt"
OUT = os.path.join(REPO, "paper"); os.makedirs(OUT, exist_ok=True)
NONLATIN = {"ar", "bn", "el", "fa", "he", "hi", "ja", "ko", "ru", "th", "uk", "zh", "te", "mi"}
torch.manual_seed(0)


def wcsv(name, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"  wrote paper/{name} ({len(rows)} rows)", flush=True)


def mk_head(state, embed, hidden, bits):
    h = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval(); h.load_state_dict(state); return h


def z_full(head, x):
    return F.normalize(head.batch_norms[-1](head.hash_head(x)[:, :1024]), p=2, dim=1)


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


def hamming(q, g, b=1024):
    return (b - q[:, :b] @ g[:, :b].t()) / 2


def topk_success(code, gal, labels, k=10):
    d = hamming(code, gal); order = d.argsort(1)
    rel = (labels[order] == labels[:, None])
    return (rel[:, :k].sum(1) > 0).cpu().numpy()


# ---------------- load ----------------
ck = torch.load(HEAD_PATH, map_location="cpu")
bits, embed, hidden = [int(b) for b in ck["bits"]], ck["embed"], ck["hidden"]
img_h = mk_head(ck["img_h"], embed, hidden, bits); txt_h = mk_head(ck["txt_h"], embed, hidden, bits)
EC = torch.load(EC_PATH, map_location="cpu")["test"]
img = F.normalize(EC["img"].float().to(dev), dim=1); true_txt = F.normalize(EC["txt"].float().to(dev), dim=1)
caps = [str(c) for c in EC["captions"]]; ids = EC["ids"]
labels = (ids if torch.is_tensor(ids) else torch.tensor([int(x) for x in ids])).to(dev)
N = img.shape[0]

with torch.no_grad():
    pred = run_distill(caps)
    e5 = F.normalize(torch.load(E5_EN, map_location="cpu").float().to(dev), dim=1)
    st = torch.load(TXTH_E5, map_location="cpu"); st = st["txt_h"] if isinstance(st, dict) and "txt_h" in st else st
    ha_head = mk_head(st, e5.shape[1], hidden, bits)
    gz = z_full(img_h, img); gcode = torch.sign(gz)
    Z = {"ceiling": z_full(txt_h, true_txt), "distillation": z_full(txt_h, pred), "head-adapt": z_full(ha_head, e5)}
    CODE = {k: torch.sign(v) for k, v in Z.items()}
true_code = CODE["ceiling"]
print(f"D: N={N}", flush=True)

# =================== D1 systematic bit flips (vs ceiling true_code) ===================
flip_rows = []
for name in ("distillation", "head-adapt"):
    flip = (CODE[name] != true_code)                       # (N,1024)
    perbit = flip.float().mean(0).cpu().numpy()            # flip frac per bit
    azc = Z[name].abs()                                    # |z| of this path
    confident = (flip & (azc > azc.median())).float().mean().item()  # flip AND large |z|
    for i in range(1024):
        flip_rows.append({"path": name, "bit": i, "flip_frac": round(float(perbit[i]), 5)})
    print(f"   {name}: mean flip {flip.float().mean():.4f} | confident-flip frac {confident:.4f} | "
          f"per-bit flip[min/med/max] {perbit.min():.3f}/{np.median(perbit):.3f}/{perbit.max():.3f}", flush=True)
wcsv("anatomy_D_bitflip.csv", flip_rows)

# =================== D2 success vs failure features ===================
succ = {name: topk_success(CODE[name], gcode, labels) for name in CODE}
caplen = np.array([len(c.split()) for c in caps])
feat_rows, pq_rows = [], []
samp = np.random.RandomState(0).choice(N, 1500, replace=False)
for name in CODE:
    s = succ[name]
    paircos = (F.cosine_similarity({"ceiling": true_txt, "distillation": pred, "head-adapt": None}[name], img, dim=1).cpu().numpy()
               if name != "head-adapt" else np.full(N, np.nan))
    ph = hamming(CODE[name], gcode)[torch.arange(N), torch.arange(N)].cpu().numpy()
    meanabsz = Z[name].abs().mean(1).cpu().numpy()
    feats = {"caplen": caplen, "pair_cosine": paircos, "pair_hamming": ph, "mean_absz": meanabsz}
    for fname, fv in feats.items():
        ok = np.isfinite(fv)
        feat_rows.append({"path": name, "feature": fname,
                          "success_mean": round(float(np.nanmean(fv[s & ok])), 4) if (s & ok).any() else None,
                          "fail_mean": round(float(np.nanmean(fv[~s & ok])), 4) if (~s & ok).any() else None,
                          "n_fail": int((~s).sum())})
    for i in samp:
        pq_rows.append({"path": name, "qidx": int(i), "success": int(s[i]), "caplen": int(caplen[i]),
                        "pair_cosine": round(float(paircos[i]), 4) if np.isfinite(paircos[i]) else "",
                        "pair_hamming": round(float(ph[i]), 1), "mean_absz": round(float(meanabsz[i]), 5)})
    print(f"   {name}: R@10 succ {100*s.mean():.2f}  fail caplen {caplen[~s].mean():.1f} vs succ {caplen[s].mean():.1f}", flush=True)
wcsv("anatomy_D_success_features.csv", feat_rows)
wcsv("anatomy_D_perquery.csv", pq_rows)

# failure overlap across the 3 paths
allfail = (~succ["ceiling"]) & (~succ["distillation"]) & (~succ["head-adapt"])
anyfail = (~succ["ceiling"]) | (~succ["distillation"]) | (~succ["head-adapt"])
wcsv("anatomy_D_fail_overlap.csv", [
    {"set": "fail_all_3paths", "count": int(allfail.sum())},
    {"set": "fail_any_path", "count": int(anyfail.sum())},
    {"set": "fail_ceiling", "count": int((~succ["ceiling"]).sum())},
    {"set": "fail_distillation", "count": int((~succ["distillation"]).sum())},
    {"set": "fail_head-adapt", "count": int((~succ["head-adapt"]).sum())},
])

# =================== E multilingual (XM3600, ceiling path) ===================
xm = torch.load(XM, map_location="cpu")
xi = F.normalize(xm["img_emb"].float().to(dev), dim=1)
langs = list(xm["per_lang"].keys())
with torch.no_grad():
    xgz = z_full(img_h, xi); xgcode = torch.sign(xgz)
Nx = xi.shape[0]
print(f"E: XM3600 imgs={Nx} langs={langs}", flush=True)
lang_rows = []
lang_codes = {}
for lg in langs:
    pl = xm["per_lang"][lg]
    te = F.normalize(pl["text_emb"].float().to(dev), dim=1)
    gold = torch.tensor([int(g) for g in pl["gold"]], device=dev)   # text -> image idx
    with torch.no_grad():
        zc = torch.sign(z_full(txt_h, te))
    lang_codes[lg] = zc.cpu().numpy().astype(np.int8)
    d = hamming(zc, xgcode)                                          # (Ntext, Nimg)
    ph = d[torch.arange(te.shape[0]), gold].cpu().numpy()           # paired text-image Hamming
    order = d.argsort(1); rel = (order == gold[:, None])
    r10 = (rel[:, :10].sum(1) > 0).float().mean().item() * 100
    lang_rows.append({"lang": lg, "script": "non-latin" if lg in NONLATIN else "latin", "n_text": int(te.shape[0]),
                      "pair_hamming_med": round(float(np.median(ph)), 1),
                      "pair_hamming_mean": round(float(ph.mean()), 1), "r10": round(r10, 2)})
    print(f"   {lg:>3} ({'NL' if lg in NONLATIN else 'L '}): pairHam med {np.median(ph):.0f} R@10 {r10:.1f}", flush=True)
wcsv("anatomy_E_lang_hamming.csv", sorted(lang_rows, key=lambda r: -r["pair_hamming_med"]))

# int8 vs fp32 bit-flip sensitivity (COCO ceiling)
def q_int8(x):
    s = x.abs().max() / 127.0
    return torch.round(x / s).clamp(-127, 127) * s
with torch.no_grad():
    z_q = z_full(txt_h, F.normalize(q_int8(true_txt), dim=1))
flip_int8 = (torch.sign(z_q) != true_code).float().mean(0).cpu().numpy()
wcsv("anatomy_E_int8_flip.csv", [{"bit": i, "flip_frac": round(float(flip_int8[i]), 5)} for i in range(1024)])
print(f"   int8 flip: mean {flip_int8.mean():.4f} max {flip_int8.max():.4f}", flush=True)

# arrays for lang scatter (sample langs incl non-latin)
samp_langs = [l for l in ["en", "de", "es", "ar", "ko", "ru", "th", "zh", "hi", "ja", "el", "bn"] if l in langs][:10]
np.savez_compressed(os.path.join(OUT, "anatomy_de_arrays.npz"),
                    xm_img=xi.cpu().numpy().astype(np.float16),
                    samp_langs=np.array(samp_langs),
                    **{f"xtxt_{l}": F.normalize(xm['per_lang'][l]['text_emb'].float(), dim=1).numpy().astype(np.float16) for l in samp_langs},
                    **{f"lcode_{l}": lang_codes[l] for l in samp_langs})
print("ANATOMY_DE_DONE", flush=True)
