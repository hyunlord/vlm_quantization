"""Evaluate the mix-ratio sweep at 1024-bit (I2T) on COCO test 5K, gold metrics,
ordered by COCO:extra step ratio. Heads share the combined-974K pool; only OI_CAP
(per-epoch extra steps) differs. baseline = no extra data; float = ceiling.
"""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti = EC["test"]["img"].float(); tt = EC["test"]["txt"].float()
ids = EC["test"]["ids"]; ids = torch.tensor(np.asarray(ids.tolist() if torch.is_tensor(ids) else ids))
N = ti.shape[0]; ti_n = F.normalize(ti, dim=1); tt_n = F.normalize(tt, dim=1); ar = torch.arange(N)
COCO_STEPS = 113287 / 512
cat2idx = {}; img_cats = {}
for f in ["data/coco/annotations/instances_val2014.json", "data/coco/annotations/instances_train2014.json"]:
    d = json.load(open(f))
    for c in d["categories"]: cat2idx.setdefault(c["id"], len(cat2idx))
    for a in d["annotations"]: img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
mh = torch.zeros(N, len(cat2idx))
for i, cid in enumerate(ids.tolist()):
    for c in img_cats.get(cid, ()): mh[i, cat2idx[c]] = 1.0
rel = (mh @ mh.t() > 0); gain = mh @ mh.t()
disc = 1.0 / torch.log2(torch.arange(2, 12).float())
idcg = (gain.sort(1, descending=True).values[:, :10] * disc).sum(1).clamp(min=1e-8)
def hamming(q, db): return (q.size(1) - q @ db.t()) / 2
def metrics(scores, smaller):
    order = scores.argsort(1, descending=not smaller)
    ranks = ((order == ar[:, None]).float().argmax(1) + 1).float()
    topk = order[:, :1000]; r = rel.gather(1, topk).float()
    prec = torch.cumsum(r, 1) / torch.arange(1, 1001).float(); nr = r.sum(1).clamp(min=1)
    g10 = gain.gather(1, order[:, :10]); ndcg = ((g10 * disc).sum(1) / idcg).mean().item()
    return {"R@1": round((ranks <= 1).float().mean().item()*100, 2), "R@10": round((ranks <= 10).float().mean().item()*100, 2),
            "MRR": round((1/ranks).mean().item()*100, 2), "mAP@1000": round(((prec*r).sum(1)/nr).mean().item()*100, 2),
            "NDCG@10": round(ndcg*100, 2)}
def head_codes(path):
    h = torch.load(path, map_location="cpu"); b = h["bits"]; bi = b.index(1024) if 1024 in b else len(b)-1
    ih = NestedHashLayer(h["embed"], h["hidden"], b, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], b, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    with torch.no_grad():
        return ih(ti_n)[bi]["binary"].float(), th(tt_n)[bi]["binary"].float()

# (label, path, cap)  cap=0 baseline(no extra), -1 float, else extra steps cap
RUNS = [
    ("float (ceiling)", None, -1),
    ("baseline (no extra)", "/tmp/k1024_coco.pt", 0),
    ("cap 113K (1:1)", "/tmp/sweep_c113287_coco_oi_rkd_crovca.pt", 113287),
    ("cap 226K (1:2)", "/tmp/sweep_c226574_coco_oi_rkd_crovca.pt", 226574),
    ("cap 403K (1:3.6)", "/tmp/k1024_974c_coco_oi_rkd_crovca.pt", 403280),
    ("cap 605K (1:5.3)", "/tmp/sweep_c604920_coco_oi_rkd_crovca.pt", 604920),
    ("uncapped 974K (1:8.6)", "/tmp/k1024_974u_coco_oi_rkd_crovca.pt", 974271),
]
print(f"{'setting':24} {'ratio':>7} | {'R@1':>6} {'R@10':>6} {'MRR':>6} {'mAP@1k':>7} {'NDCG10':>7}", flush=True)
print("-"*78)
for label, path, cap in RUNS:
    if cap == -1: m = metrics(ti_n @ tt_n.t(), False); ratio = "-"
    elif not os.path.exists(path): print(f"{label:24} {'MISSING':>7}"); continue
    else:
        ic, tc = head_codes(path); m = metrics(hamming(ic, tc), True)
        ratio = "1:0" if cap == 0 else f"1:{(cap/512)/COCO_STEPS:.1f}"
    print(f"{label:24} {ratio:>7} | {m['R@1']:6} {m['R@10']:6} {m['MRR']:6} {m['mAP@1000']:7} {m['NDCG@10']:7}", flush=True)
print("SWEEP_EVAL_DONE", flush=True)
