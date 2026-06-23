"""Anatomy F (neutral deeper cuts):
 F1 distill-collapse: cumulative explained-variance (image/ceiling/distillation), and
    residual energy of distillation outside ceiling's top-k PC subspace.
 F2 low-resource language geometry (XM3600, ceiling path): per-language text-code centroid
    Hamming-distance to image-code centroid, per-language code bit-entropy, pair-Hamming spread vs R@10.
Dumps paper/anatomy_F*.csv. Run: .venv/bin/python scripts/anatomy_f.py
"""
from __future__ import annotations
import csv, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(REPO, "paper")
HEAD_PATH="/tmp/ft_ko_113.pt"; EC_PATH="/tmp/emb_cache.pt"; DISTILL="/tmp/distill_e5.pt"; XM="/tmp/xm_so400m_lc.pt"
NONLATIN = {"ar","bn","el","fa","he","hi","ja","ko","ru","th","uk","zh","te","mi"}


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
caps = [str(c) for c in EC["captions"]]

# ---------- F1 distill collapse ----------
with torch.no_grad():
    pred = run_distill(caps)
srcs = {"image": img, "ceiling": true_txt, "distillation": pred}
cev_rows = []
basis = {}
for name, X in srcs.items():
    Xc = (X - X.mean(0, keepdim=True)).float()
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    ev = (S**2 / (S**2).sum()).cpu().numpy(); cum = np.cumsum(ev)
    basis[name] = Vt
    for k in [1,2,5,10,20,50,100,200,500,1024]:
        if k <= len(cum):
            cev_rows.append({"source": name, "k": k, "cum_var_frac": round(float(cum[k-1]), 5)})
    print(f"   {name}: dims for 90% var = {int(np.searchsorted(cum,0.9)+1)}, 99% = {int(np.searchsorted(cum,0.99)+1)}", flush=True)
wcsv("anatomy_F1_cumvar.csv", cev_rows)

# residual energy of distillation outside ceiling top-k subspace (and vice versa)
res_rows = []
for a, b in [("distillation", "ceiling"), ("ceiling", "distillation"), ("image", "ceiling")]:
    Xa = (srcs[a] - srcs[a].mean(0, keepdim=True)).float()
    tot = (Xa**2).sum().item()
    Vb = basis[b]  # rows are PCs of b
    for k in [5,10,20,50,100,200]:
        proj_energy = (Xa @ Vb[:k].t()).pow(2).sum().item()
        res_rows.append({"projected": a, "onto_basis_of": b, "k": k,
                         "energy_in_subspace_frac": round(proj_energy/tot, 5),
                         "residual_frac": round(1 - proj_energy/tot, 5)})
    print(f"   {a} energy in {b} top-50 PCs: {round((Xa@Vb[:50].t()).pow(2).sum().item()/tot,4)}", flush=True)
wcsv("anatomy_F1_residual.csv", res_rows)

# ---------- F2 low-resource language geometry ----------
xm = torch.load(XM, map_location="cpu")
xi = F.normalize(xm["img_emb"].float().to(dev), dim=1)
with torch.no_grad():
    img_code = torch.sign(zf(img_h, xi))
img_centroid = img_code.mean(0)  # mean ±1 -> per-bit balance vector
Nx = xi.shape[0]


def ham(q, g):
    return (1024 - q @ g.t()) / 2
f2 = []
for lg in xm["per_lang"]:
    pl = xm["per_lang"][lg]; te = F.normalize(pl["text_emb"].float().to(dev), dim=1)
    gold = torch.tensor([int(g) for g in pl["gold"]], device=dev)
    with torch.no_grad():
        tc = torch.sign(zf(txt_h, te))
    d = ham(tc, img_code); ph = d[torch.arange(te.shape[0]), gold].cpu().numpy()
    order = d.argsort(1); r10 = ((order == gold[:, None])[:, :10].sum(1) > 0).float().mean().item()*100
    txt_centroid = tc.mean(0)
    centroid_ham = float(((1024 - txt_centroid @ img_centroid) / 2).item())  # soft Hamming between centroids
    pos = ((tc > 0).float().mean(0)).cpu().numpy(); p = np.clip(pos, 1e-6, 1-1e-6)
    bitH = float((-(p*np.log2(p)+(1-p)*np.log2(1-p))).sum())
    # text-emb effective dim
    Xc = (te - te.mean(0, keepdim=True)).float(); s = torch.linalg.svdvals(Xc).cpu().numpy()**2
    pr = float((s.sum()**2)/(s**2).sum())
    f2.append({"lang": lg, "script": "non-latin" if lg in NONLATIN else "latin",
               "r10": round(r10,2), "pair_ham_med": round(float(np.median(ph)),1),
               "pair_ham_std": round(float(ph.std()),1), "centroid_softham": round(centroid_ham,2),
               "code_bit_entropy": round(bitH,1), "txtemb_effdim": round(pr,1)})
    print(f"   {lg:>3}: R@10 {r10:5.1f} centroidH {centroid_ham:6.1f} codeH {bitH:6.1f} effdim {pr:5.1f}", flush=True)
wcsv("anatomy_F2_lang_geometry.csv", sorted(f2, key=lambda r: r["r10"]))
print("ANATOMY_F_DONE", flush=True)
