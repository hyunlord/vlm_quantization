"""Korean T2I retrieval eval — measures whether Korean queries work on a trained
hash head, and the gap vs English on the SAME 5K COCO test images.

Gold: paired (Korean caption ↔ its own COCO image), Hamming search over 5K images.
Reports R@1/5/10, MRR, MedR at bit prefixes 64/256/1024, for English (reference)
and Korean, per head. No training data touched (test split only → no leak).

Env: HEADS (comma list of label=path), BITS_EVAL(64,256,1024)
Inputs: /tmp/coco_ko_test.pt (Korean test caps + ids), /tmp/emb_cache.pt (test img/txt/ids)
"""
from __future__ import annotations
import os, sys, json
import torch, torch.nn.functional as F
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
BITS_EVAL = [int(x) for x in os.environ.get("BITS_EVAL", "64,256,1024").split(",")]
DEFAULT_HEADS = (
    "baseline(coco)=/tmp/k1024_coco.pt,"
    "best(1:2 mix)=/tmp/sweep_c226574_coco_oi_rkd_crovca.pt,"
    "coco+cc12m+oi(974c)=/tmp/k1024_974c_coco_oi_rkd_crovca.pt"
)
HEADS = []
for spec in os.environ.get("HEADS", DEFAULT_HEADS).split(","):
    spec = spec.strip()
    if "=" in spec:
        lab, p = spec.split("=", 1); HEADS.append((lab, p))

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
te_img = EC["test"]["img"].float(); te_en = EC["test"]["txt"].float()
te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
ko_txt = KO["txt_emb"].float(); ko_ids = [int(x) for x in KO["ids"].tolist()]
assert ko_ids == te_ids, "Korean test ids must align to emb_cache test order"
labels = torch.tensor(te_ids)
print(f"test images {te_img.shape[0]:,} | EN caps {te_en.shape[0]:,} | KO caps {ko_txt.shape[0]:,}", flush=True)


def nrm(x):
    return F.normalize(x, dim=1)


def hdist(q, db):
    return (q.size(1) - q @ db.t()) / 2  # Hamming from ±1 codes


def t2i_metrics(tcodes, icodes):
    d = hdist(tcodes, icodes)                       # [Nt, Ni]
    order = d.argsort(dim=1, descending=False)      # nearest image first
    gold = (labels[order] == labels[:, None])       # paired gold
    rank = gold.float().argmax(dim=1) + 1           # rank of first (only) gold
    out = {f"R@{k}": round((gold[:, :k].any(dim=1)).float().mean().item() * 100, 2) for k in (1, 5, 10)}
    out["MRR"] = round((1.0 / rank.float()).mean().item(), 4)
    out["MedR"] = int(rank.median().item())
    return out


def load_head(path):
    ck = torch.load(path, map_location="cpu")
    bits, hidden, embed = ck["bits"], ck["hidden"], ck["embed"]
    ni = ck.get("norm_in", 1)
    ih = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval()
    th = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval()
    ih.load_state_dict(ck["img_h"]); th.load_state_dict(ck["txt_h"])
    return ih, th, bits, ni


results = {}
for lab, path in HEADS:
    if not os.path.exists(path):
        print(f"!! missing {lab} {path}", flush=True); continue
    ih, th, bits, ni = load_head(path)
    prep = nrm if ni else (lambda x: x)
    img_in, en_in, ko_in = prep(te_img).to(dev), prep(te_en).to(dev), prep(ko_txt).to(dev)
    with torch.no_grad():
        io, eo, ko_o = ih(img_in), th(en_in), th(ko_in)
    res = {}
    for b in BITS_EVAL:
        if b not in bits:
            continue
        bi = bits.index(b)
        ic = io[bi]["binary"].cpu().float()
        ec = eo[bi]["binary"].cpu().float()
        kc = ko_o[bi]["binary"].cpu().float()
        res[b] = {"EN": t2i_metrics(ec, ic), "KO": t2i_metrics(kc, ic)}
    results[lab] = res
    print(f"\n=== {lab} ({path.split('/')[-1]}) norm_in={ni} ===", flush=True)
    for b in BITS_EVAL:
        if b in res:
            en, ko = res[b]["EN"], res[b]["KO"]
            print(f"  {b:>4}bit  EN R@1/5/10 {en['R@1']:5}/{en['R@5']:5}/{en['R@10']:5} MRR {en['MRR']} | "
                  f"KO R@1/5/10 {ko['R@1']:5}/{ko['R@5']:5}/{ko['R@10']:5} MRR {ko['MRR']} | "
                  f"gap@10 {round(en['R@10']-ko['R@10'],2)}", flush=True)

json.dump(results, open("/tmp/korean_eval_results.json", "w"), indent=2)
print("\nKOREAN_EVAL_DONE -> /tmp/korean_eval_results.json", flush=True)
