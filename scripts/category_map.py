"""Category-based mAP@K — the deep-hashing literature gold (relevance = share >=1
of 80 COCO categories), complementary to pair-R@K. Rewards semantically-similar
retrievals (not just the exact pair). COCO test 5K, cross-modal I2T & T2I, 256-bit.

AP@K per query = sum_{k<=K}(P@k * rel_k) / max(1, #relevant in top-K); mAP = mean.
Each head uses its as-trained input convention (demo=raw, others=L2norm). CPU.
Out: /tmp/category_map.json
"""
from __future__ import annotations
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

K = int(os.environ.get("MAP_K", "1000"))
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti_raw = EC["test"]["img"].float(); tt_raw = EC["test"]["txt"].float()
ids = EC["test"]["ids"]; ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
N = ti_raw.shape[0]
ti_n = F.normalize(ti_raw, dim=1); tt_n = F.normalize(tt_raw, dim=1)

# build 80-cat multihot from instances annotations (train+val2014 cover Karpathy test)
cat2idx = {}; img_cats = {}
for f in ["data/coco/annotations/instances_val2014.json", "data/coco/annotations/instances_train2014.json"]:
    d = json.load(open(f))
    for c in d["categories"]:
        cat2idx.setdefault(c["id"], len(cat2idx))
    for a in d["annotations"]:
        img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
C = len(cat2idx)
mh = torch.zeros(N, C)
miss = 0
for i, cid in enumerate(ids):
    cs = img_cats.get(cid)
    if not cs: miss += 1; continue
    for c in cs: mh[i, cat2idx[c]] = 1.0
print(f"N={N} cats={C} | imgs w/o category labels={miss}", flush=True)
rel_all = (mh @ mh.t() > 0)   # (N,N) bool relevance
avg_rel = rel_all.float().sum(1).mean().item()
print(f"avg #relevant per query = {avg_rel:.0f} / {N}", flush=True)


def hamming(q, db): return (q.size(1) - q @ db.t()) / 2


def mapk(scores, smaller):
    order = scores.argsort(dim=1, descending=not smaller)
    aps = []
    ar = torch.arange(1, K + 1, dtype=torch.float32)
    for i in range(N):
        rel = rel_all[i, order[i, :K]].float()
        nr = rel.sum()
        if nr == 0: aps.append(0.0); continue
        prec = torch.cumsum(rel, 0) / ar
        aps.append(((prec * rel).sum() / nr).item())
    return round(float(np.mean(aps)) * 100, 2)


def head_codes(path, norm_in):
    h = torch.load(path, map_location="cpu")
    bits = h["bits"]; bi = bits.index(256) if 256 in bits else len(bits) - 1
    ih = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    xi = ti_n if norm_in else ti_raw; xt = tt_n if norm_in else tt_raw
    with torch.no_grad():
        return ih(xi)[bi]["binary"].float(), th(xt)[bi]["binary"].float()


RUNS = [
    ("float (ceiling)", None, True),
    ("bit-old demo (raw)", "/tmp/demo_hashheads.pt", False),
    ("coco baseline", "/tmp/mixctrl_head_A_coco.pt", True),
    ("CC12M 403K (A)", "/tmp/mixctrl_head_A_coco_oi_rkd_crovca.pt", True),
    ("rich 974K (B)", "/tmp/mixctrl_head_B_coco_oi_rkd_crovca.pt", True),
    ("ablate NORM", "/tmp/ablate_head_norm.pt", True),
    ("ablate RAW (raw)", "/tmp/ablate_head_raw.pt", False),
]
res = {}
for label, path, norm_in in RUNS:
    if path is None:
        i2t = mapk(ti_n @ tt_n.t(), False); t2i = mapk(tt_n @ ti_n.t(), False)
    else:
        if not os.path.exists(path): print(f"  SKIP {label}", flush=True); continue
        ic, tc = head_codes(path, norm_in)
        i2t = mapk(hamming(ic, tc), True); t2i = mapk(hamming(tc, ic), True)
    res[label] = {"I2T_mAP@%d" % K: i2t, "T2I_mAP@%d" % K: t2i}
    print(f"  {label:26}: I2T mAP@{K} {i2t:6}  T2I mAP@{K} {t2i:6}", flush=True)
json.dump({"K": K, "avg_rel": avg_rel, "results": res}, open("/tmp/category_map.json", "w"), indent=2)
print("CATMAP_DONE", flush=True)
