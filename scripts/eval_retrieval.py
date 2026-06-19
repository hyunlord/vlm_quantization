"""PROPER gold re-baseline: COCO test 5K cross-modal retrieval against the TRUE
paired item (not float-mimicry). Reports R@1/5/10, MRR, MedianRank for I2T & T2I,
for float (ceiling) and every saved hash head, at the 256-bit prefix.

Each head uses its AS-TRAINED input convention (demo=raw pooled emb; the
train_broaden/mixctrl/ablate heads=L2-normalized). Forced CPU (5K is trivial).
Out: /tmp/retrieval_metrics.json + console table.
"""
from __future__ import annotations
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti_raw = EC["test"]["img"].float(); tt_raw = EC["test"]["txt"].float()
ids = EC["test"]["ids"]
if not torch.is_tensor(ids): ids = torch.tensor(np.asarray(ids))
N = ti_raw.shape[0]
ti_n = F.normalize(ti_raw, dim=1); tt_n = F.normalize(tt_raw, dim=1)
print(f"COCO test: N={N} (1:1 image-caption); ids unique={len(torch.unique(ids))}", flush=True)


def retr_metrics(scores, smaller_better):
    """scores (Nq, Ng); true relevant = same id. Returns R@1/5/10, MRR, MedR."""
    order = scores.argsort(dim=1, descending=not smaller_better)
    rel = (ids[order] == ids[:, None])                  # (Nq, Ng) bool, relevance mask
    # rank (1-indexed) of FIRST relevant gallery item
    first = rel.float().argmax(dim=1) + 1
    first = first.float()
    rk = lambda k: (rel[:, :k].sum(1) > 0).float().mean().item()
    return {"R@1": round(rk(1)*100, 2), "R@5": round(rk(5)*100, 2), "R@10": round(rk(10)*100, 2),
            "MRR": round((1.0/first).mean().item()*100, 2), "MedR": int(first.median().item())}


def hamming(q, db): return (q.size(1) - q @ db.t()) / 2


def head_codes(path, norm_in):
    h = torch.load(path, map_location="cpu")
    bits = h["bits"]; bi = bits.index(256) if 256 in bits else len(bits) - 1
    ih = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    xi = ti_n if norm_in else ti_raw; xt = tt_n if norm_in else tt_raw
    with torch.no_grad():
        ic = ih(xi)[bi]["binary"].float(); tc = th(xt)[bi]["binary"].float()
    return ic, tc, bits[bi]


# (label, path|None, norm_in)
RUNS = [
    ("float (ceiling)", None, True),
    ("bit-old demo (raw, 256/1024)", "/tmp/demo_hashheads.pt", False),
    ("coco baseline (A_coco)", "/tmp/mixctrl_head_A_coco.pt", True),
    ("CC12M 403K (A)", "/tmp/mixctrl_head_A_coco_oi_rkd_crovca.pt", True),
    ("rich 974K capped (B)", "/tmp/mixctrl_head_B_coco_oi_rkd_crovca.pt", True),
    ("ablate NORM", "/tmp/ablate_head_norm.pt", True),
    ("ablate RAW (raw)", "/tmp/ablate_head_raw.pt", False),
]

results = {}
for label, path, norm_in in RUNS:
    if path is None:
        i2t = retr_metrics(ti_n @ tt_n.t(), smaller_better=False)
        t2i = retr_metrics(tt_n @ ti_n.t(), smaller_better=False)
        nbit = "float-emb"
    else:
        if not os.path.exists(path):
            print(f"  SKIP {label}: {path} missing", flush=True); continue
        ic, tc, nbit = head_codes(path, norm_in)
        i2t = retr_metrics(hamming(ic, tc), smaller_better=True)   # img query -> text gallery
        t2i = retr_metrics(hamming(tc, ic), smaller_better=True)   # text query -> img gallery
    results[label] = {"bits": str(nbit), "norm_in": norm_in, "I2T": i2t, "T2I": t2i}
    print(f"  done: {label}", flush=True)

# console table
hdr = f"{'run':32} {'bit':9} | I2T R@1/5/10 MRR MedR        | T2I R@1/5/10 MRR MedR"
print("\n" + hdr); print("-"*len(hdr))
for label, r in results.items():
    a, b = r["I2T"], r["T2I"]
    print(f"{label:32} {r['bits']:9} | "
          f"{a['R@1']:5}/{a['R@5']:5}/{a['R@10']:5} {a['MRR']:5} {a['MedR']:4}   | "
          f"{b['R@1']:5}/{b['R@5']:5}/{b['R@10']:5} {b['MRR']:5} {b['MedR']:4}")
json.dump(results, open("/tmp/retrieval_metrics.json", "w"), indent=2)
print("\nsaved /tmp/retrieval_metrics.json")
print("RETR_EVAL_DONE", flush=True)
