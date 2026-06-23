"""Anatomy D-deep — (1) flip-fraction vs |z| quantile per path (confident vs boundary flips),
(2) per-query separation gap (nearest-wrong - pair) by success/failure, per path.
Neutral. Dumps paper/anatomy_Ddeep_*.csv. Run: .venv/bin/python scripts/anatomy_d_deep.py
"""
from __future__ import annotations
import csv, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(REPO, "paper")
HEAD_PATH="/tmp/ft_ko_113.pt"; EC_PATH="/tmp/emb_cache.pt"; DISTILL="/tmp/distill_e5.pt"; TXTH_E5="/tmp/txt_h_e5.pt"; E5_EN="/tmp/e5_test_en.pt"


def wcsv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote paper/{name} ({len(rows)})", flush=True)


def mk(state, e, h, b):
    m = NestedHashLayer(e, h, b, 0.0).to(dev).eval(); m.load_state_dict(state); return m


def zf(head, x):
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


ck = torch.load(HEAD_PATH, map_location="cpu"); bits, embed, hidden = [int(b) for b in ck["bits"]], ck["embed"], ck["hidden"]
img_h = mk(ck["img_h"], embed, hidden, bits); txt_h = mk(ck["txt_h"], embed, hidden, bits)
EC = torch.load(EC_PATH, map_location="cpu")["test"]
img = F.normalize(EC["img"].float().to(dev), dim=1); true_txt = F.normalize(EC["txt"].float().to(dev), dim=1)
caps = [str(c) for c in EC["captions"]]; ids = EC["ids"]
labels = (ids if torch.is_tensor(ids) else torch.tensor([int(x) for x in ids])).to(dev); N = img.shape[0]
with torch.no_grad():
    pred = run_distill(caps); e5 = F.normalize(torch.load(E5_EN, map_location="cpu").float().to(dev), dim=1)
    st = torch.load(TXTH_E5, map_location="cpu"); st = st["txt_h"] if isinstance(st, dict) and "txt_h" in st else st
    ha = mk(st, e5.shape[1], hidden, bits)
    gz = zf(img_h, img); gcode = torch.sign(gz)
    Z = {"ceiling": zf(txt_h, true_txt), "distillation": zf(txt_h, pred), "head-adapt": zf(ha, e5)}
    CODE = {k: torch.sign(v) for k, v in Z.items()}
true_code = CODE["ceiling"]

# (1) flip fraction vs |z| decile (per path, flips measured vs ceiling true_code)
rows1 = []
for name in ("distillation", "head-adapt"):
    az = Z[name].abs().flatten().cpu().numpy()
    flip = (CODE[name] != true_code).flatten().cpu().numpy()
    q = np.quantile(az, np.linspace(0, 1, 11))
    for i in range(10):
        m = (az >= q[i]) & (az <= q[i+1] if i == 9 else az < q[i+1])
        rows1.append({"path": name, "absz_decile": i+1, "absz_lo": round(float(q[i]), 5), "absz_hi": round(float(q[i+1]), 5),
                      "flip_frac": round(float(flip[m].mean()), 5), "n": int(m.sum())})
    print(f"   {name}: flip by |z| decile (lo->hi): {[round(float(flip[(az>=q[i])&(az<q[i+1])].mean()),3) for i in range(10)]}", flush=True)
wcsv("anatomy_Ddeep_flip_vs_absz.csv", rows1)

# (2) per-query separation gap (nearest-wrong - pair) by success/fail, per path
def ham(q, g):
    return (1024 - q @ g.t()) / 2
rows2 = []
for name in CODE:
    d = ham(CODE[name], gcode)
    pair = d[torch.arange(N), torch.arange(N)]
    dd = d.clone(); dd[torch.arange(N), torch.arange(N)] = 1e9
    nwrong = dd.min(1).values
    gap = (nwrong - pair).cpu().numpy()
    order = d.argsort(1); rel = (labels[order] == labels[:, None])
    succ = (rel[:, :10].sum(1) > 0).cpu().numpy()
    for grp, mask in [("success", succ), ("fail", ~succ)]:
        g = gap[mask]
        h, e = np.histogram(g, bins=60, range=(-200, 200))
        for c, e0 in zip(h, e[:-1]):
            rows2.append({"path": name, "group": grp, "gap_bin": int(e0), "count": int(c)})
        print(f"   {name} {grp}: gap median {np.median(g):.0f} mean {g.mean():.0f} (n={mask.sum()})", flush=True)
wcsv("anatomy_Ddeep_gap.csv", rows2)
print("ANATOMY_DDEEP_DONE", flush=True)
