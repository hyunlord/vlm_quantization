"""Hash-NATIVE metrics + code-quality diagnostics (complements eval_all.py).
Every head x every Matryoshka prefix, COCO test 5K, category (80-cat) relevance.

Hash-lookup metrics (float baseline N/A — these prove O(1) table lookup works):
  P@H<=2, R@H<=2 (precision/recall within Hamming radius 2), ball coverage
  NDCG@1000 (graded, gain = #shared categories)
Code-quality diagnostics (on image codes):
  bit_balance  = mean_k |mean_i h_ik|            (0 = perfectly balanced)
  bit_indep    = ||(1/N)HᵀH − I||_F / K          (0 = uncorrelated bits)
  entropy      = mean_k H(p_k) bits/bit          (1 = max info per bit)
  quant_err    = mean_ik (1 − |z_ik|)            (0 = hard ±1 commitment)
Out: /tmp/eval_hashing.json
"""
from __future__ import annotations
import json, os, sys, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti_raw = EC["test"]["img"].float(); tt_raw = EC["test"]["txt"].float()
ids = EC["test"]["ids"]; ids = torch.tensor(np.asarray(ids.tolist() if torch.is_tensor(ids) else ids))
N = ti_raw.shape[0]
ti_n = F.normalize(ti_raw, dim=1); tt_n = F.normalize(tt_raw, dim=1)

cat2idx = {}; img_cats = {}
for f in ["data/coco/annotations/instances_val2014.json", "data/coco/annotations/instances_train2014.json"]:
    d = json.load(open(f))
    for c in d["categories"]: cat2idx.setdefault(c["id"], len(cat2idx))
    for a in d["annotations"]: img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
mh = torch.zeros(N, len(cat2idx))
for i, cid in enumerate(ids.tolist()):
    for c in img_cats.get(cid, ()): mh[i, cat2idx[c]] = 1.0
gain_all = mh @ mh.t()
rel_cat = (gain_all > 0)
tot_rel = rel_cat.float().sum(1).clamp(min=1)        # per-query #relevant
disc1k = 1.0 / torch.log2(torch.arange(2, 1002).float())
idcg1k = (gain_all.sort(1, descending=True).values[:, :1000] * disc1k).sum(1).clamp(min=1e-8)
print(f"N={N} cats={len(cat2idx)}", flush=True)


def hamming(q, db): return (q.size(1) - q @ db.t()) / 2


def hball(scores):   # scores = hamming (Nq,Ng); radius 2
    inball = (scores <= 2)
    nb = inball.float().sum(1)
    relin = (inball & rel_cat).float().sum(1)
    cov = (nb > 0)
    P = (relin[cov] / nb[cov].clamp(min=1)).mean().item() if cov.any() else 0.0
    R = (relin / tot_rel).mean().item()
    return {"P@H2": round(P*100, 2), "R@H2": round(R*100, 4), "cover@H2": round(cov.float().mean().item()*100, 1)}


def ndcg1k(scores, smaller):
    order = scores.argsort(1, descending=not smaller)
    g = gain_all.gather(1, order[:, :1000])
    return round(((g * disc1k).sum(1) / idcg1k).mean().item()*100, 2)


def diag(binc, cont):
    H = binc  # (N,b) in {+1,-1}
    b = H.shape[1]
    bal = H.mean(0).abs().mean().item()
    C = (H.t() @ H) / N
    indep = (torch.norm(C - torch.eye(b)) / b).item()
    p = ((H.mean(0) + 1) / 2).clamp(1e-6, 1-1e-6)
    ent = (-(p*torch.log2(p) + (1-p)*torch.log2(1-p))).mean().item()
    qe = (1 - cont.abs()).mean().item()
    return {"bit_balance": round(bal, 4), "bit_indep": round(indep, 4),
            "entropy": round(ent, 4), "quant_err": round(qe, 4)}


def all_codes(path, norm_in):
    h = torch.load(path, map_location="cpu")
    bits = h["bits"]
    ih = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    xi = ti_n if norm_in else ti_raw; xt = tt_n if norm_in else tt_raw
    with torch.no_grad():
        io = ih(xi); to = th(xt)
    return bits, io, to


HEADS = [
    ("bit-old demo (h768)", "/tmp/demo_hashheads.pt", False),
    ("COCO baseline", "/tmp/k1024_coco.pt", True),
    ("COCO + CroVCA", "/tmp/k1024_coco_crovca.pt", True),
    ("CC12M 403K", "/tmp/k1024_coco_oi_rkd_crovca.pt", True),
    ("974K uncapped", "/tmp/k1024_974u_coco_oi_rkd_crovca.pt", True),
    ("974K capped", "/tmp/k1024_974c_coco_oi_rkd_crovca.pt", True),
    ("COCO raw-input", "/tmp/k1024_raw_coco.pt", False),
]

out = {}
for label, path, norm_in in HEADS:
    if not os.path.exists(path): print(f"SKIP {label}", flush=True); continue
    bits, io, to = all_codes(path, norm_in)
    perbit = {}
    for bi, b in enumerate(bits):
        ic = io[bi]["binary"].float(); tc = to[bi]["binary"].float()
        si2t = hamming(ic, tc); st2i = hamming(tc, ic)
        perbit[str(b)] = {
            "I2T": {**hball(si2t), "NDCG@1000": ndcg1k(si2t, True)},
            "T2I": {**hball(st2i), "NDCG@1000": ndcg1k(st2i, True)},
            "diag_img": diag(ic, io[bi]["continuous"]),
        }
    out[label] = perbit
    print(f"done {label}", flush=True)

json.dump(out, open("/tmp/eval_hashing.json", "w"), indent=2)
print("EVAL_HASHING_DONE", flush=True)
