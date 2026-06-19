"""COMPREHENSIVE evaluation: every saved head x every Matryoshka prefix x many
gold metrics x {I2T, T2I}, on COCO test 5K. Fully vectorized (fast).

Metrics
  pair-based (relevance = exact paired item; ids unique -> single relevant):
    R@1, R@5, R@10, MRR, MedR
  category-based (relevance = share >=1 of 80 COCO categories; gain = #shared):
    mAP@100, mAP@1000, NDCG@10, P@10
Each head uses its as-trained input convention (demo/ablate-raw = raw pooled emb;
others = L2-normalized). float baseline = normalized cosine (bit-independent).
Out: /tmp/eval_all.json
"""
from __future__ import annotations
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti_raw = EC["test"]["img"].float(); tt_raw = EC["test"]["txt"].float()
ids = EC["test"]["ids"]; ids = torch.tensor(np.asarray(ids.tolist() if torch.is_tensor(ids) else ids))
N = ti_raw.shape[0]
ti_n = F.normalize(ti_raw, dim=1); tt_n = F.normalize(tt_raw, dim=1)
ar = torch.arange(N)

# ---- 80-cat multihot -> relevance + gain matrices ----
cat2idx = {}; img_cats = {}
for f in ["data/coco/annotations/instances_val2014.json", "data/coco/annotations/instances_train2014.json"]:
    d = json.load(open(f))
    for c in d["categories"]: cat2idx.setdefault(c["id"], len(cat2idx))
    for a in d["annotations"]: img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
mh = torch.zeros(N, len(cat2idx))
for i, cid in enumerate(ids.tolist()):
    for c in img_cats.get(cid, ()): mh[i, cat2idx[c]] = 1.0
gain_all = mh @ mh.t()                 # (N,N) #shared categories
rel_cat = (gain_all > 0)               # (N,N) bool category relevance
ideal10 = gain_all.sort(dim=1, descending=True).values[:, :10]
disc10 = 1.0 / torch.log2(torch.arange(2, 12).float())
idcg10 = (ideal10 * disc10).sum(1).clamp(min=1e-8)
print(f"N={N} cats={len(cat2idx)} avg_rel={rel_cat.float().sum(1).mean():.0f}", flush=True)


def pair_metrics(scores, smaller):
    order = scores.argsort(dim=1, descending=not smaller)
    ranks = (order == ar[:, None]).float().argmax(1) + 1     # rank of the true paired item
    ranks = ranks.float()
    return {"R@1": round((ranks <= 1).float().mean().item()*100, 2),
            "R@5": round((ranks <= 5).float().mean().item()*100, 2),
            "R@10": round((ranks <= 10).float().mean().item()*100, 2),
            "MRR": round((1.0/ranks).mean().item()*100, 2),
            "MedR": int(ranks.median().item())}


def cat_metrics(scores, smaller):
    order = scores.argsort(dim=1, descending=not smaller)
    def mapk(K):
        topk = order[:, :K]
        rel = rel_cat.gather(1, topk).float()                # (N,K)
        prec = torch.cumsum(rel, 1) / torch.arange(1, K+1).float()
        nr = rel.sum(1).clamp(min=1)
        return round(((prec*rel).sum(1)/nr).mean().item()*100, 2)
    top10 = order[:, :10]
    g10 = gain_all.gather(1, top10)
    ndcg = ((g10 * disc10).sum(1) / idcg10).mean().item()
    p10 = rel_cat.gather(1, top10).float().mean().item()
    return {"mAP@100": mapk(100), "mAP@1000": mapk(1000),
            "NDCG@10": round(ndcg*100, 2), "P@10": round(p10*100, 2)}


def hamming(q, db): return (q.size(1) - q @ db.t()) / 2


def all_codes(path, norm_in):
    h = torch.load(path, map_location="cpu")
    bits = h["bits"]
    ih = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    xi = ti_n if norm_in else ti_raw; xt = tt_n if norm_in else tt_raw
    with torch.no_grad():
        io = ih(xi); to = th(xt)
    return bits, [o["binary"].float() for o in io], [o["binary"].float() for o in to]


# All settings trained at bits->1024 (k1024_*, hidden 384) + demo (h768->1024).
HEADS = [
    ("float (ceiling)", None, True),
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
    if path is None:
        m = {"I2T": {**pair_metrics(ti_n @ tt_n.t(), False), **cat_metrics(ti_n @ tt_n.t(), False)},
             "T2I": {**pair_metrics(tt_n @ ti_n.t(), False), **cat_metrics(tt_n @ ti_n.t(), False)}}
        out[label] = {"float": m}
        print(f"done float", flush=True); continue
    if not os.path.exists(path): print(f"SKIP {label}", flush=True); continue
    bits, ics, tcs = all_codes(path, norm_in)
    perbit = {}
    for bi, b in enumerate(bits):
        si2t = hamming(ics[bi], tcs[bi]); st2i = hamming(tcs[bi], ics[bi])
        perbit[str(b)] = {"I2T": {**pair_metrics(si2t, True), **cat_metrics(si2t, True)},
                          "T2I": {**pair_metrics(st2i, True), **cat_metrics(st2i, True)}}
    out[label] = perbit
    print(f"done {label} (bits {bits})", flush=True)

json.dump(out, open("/tmp/eval_all.json", "w"), indent=2)
print("EVAL_ALL_DONE", flush=True)
