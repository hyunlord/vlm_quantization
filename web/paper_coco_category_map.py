"""(c) COCO 80-category mAP sanity check — CMH-reviewer insurance for the
instance-vs-category pre-emption paragraph. Relevance = two images share >=1 of the
80 COCO object categories (from instances_{train,val}2014.json). We rank our deployed
1-bit Hamming codes and report category-mAP. CPU-only (5K x 5K is trivial), so it runs
alongside the GPU jobs.

Rows (deployed 1024-bit codes; image gallery = frozen ft113 img_h, shared by every text head):
  float (ceiling)            so400m cosine ranking  (semantic upper bound)
  server (so400m+ft113)      so400m text -> ft113 txt_h          T2I
  deployed offline (e5)      e5-small text -> txt_h_e5           T2I   (browser-deployable head)
  ft113 image codes          img_h(test_img)                    I2I   (shared by server & offline)

Tasks: T2I (text->image, our main direction) and I2I (image->image, self excluded).
Metrics: mAP@10 (primary), mAP@R (R = #relevant per query), mAP@100.
AP@K denominator = #relevant retrieved in top-K (same convention as scripts/category_map.py,
so /tmp/category_map.json cross-checks).

Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_coco_category_map.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only -> free to run beside GPU jobs
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

PAPER = Path(REPO) / "paper"
BIT = int(os.environ.get("CODE_BITS", "1024"))  # deployed code length
HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
E5_HEAD = os.environ.get("E5_HEAD", "/tmp/txt_h_e5.pt")

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti_raw = EC["test"]["img"].float()
tt_raw = EC["test"]["txt"].float()
ids = EC["test"]["ids"]
ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
N = ti_raw.shape[0]
ti_n = F.normalize(ti_raw, dim=1)
tt_n = F.normalize(tt_raw, dim=1)
en_caps = [str(c) for c in EC["test"]["captions"]]

# ---- 80-category multi-hot relevance (Karpathy test cocoids -> instance categories) ----
cat2idx, img_cats = {}, {}
for f in ["data/coco/annotations/instances_val2014.json", "data/coco/annotations/instances_train2014.json"]:
    d = json.load(open(f"{REPO}/{f}"))
    for c in d["categories"]:
        cat2idx.setdefault(c["id"], len(cat2idx))
    for a in d["annotations"]:
        img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
C = len(cat2idx)
mh = torch.zeros(N, C)
miss = 0
for i, cid in enumerate(ids):
    cs = img_cats.get(cid)
    if not cs:
        miss += 1
        continue
    for c in cs:
        mh[i, cat2idx[c]] = 1.0
rel_all = (mh @ mh.t() > 0)  # (N,N) bool
avg_rel = rel_all.float().sum(1).mean().item()
print(f"[catmap] N={N} cats={C} imgs_without_labels={miss} avg_relevant/query={avg_rel:.0f}", flush=True)


def hamming(q, db):
    return (q.size(1) - q @ db.t()) / 2


def mapk(scores, smaller, K, exclude_self=False):
    order = scores.argsort(dim=1, descending=not smaller)
    aps = []
    for i in range(N):
        oi = order[i]
        if exclude_self:
            oi = oi[oi != i]
        Ki = K if K > 0 else int(rel_all[i].sum().item())  # K<=0 -> mAP@R
        oi = oi[:Ki]
        if oi.numel() == 0:
            aps.append(0.0)
            continue
        rel = rel_all[i, oi].float()
        nr = rel.sum()
        if nr == 0:
            aps.append(0.0)
            continue
        prec = torch.cumsum(rel, 0) / torch.arange(1, oi.numel() + 1, dtype=torch.float32)
        aps.append(((prec * rel).sum() / nr).item())
    return round(float(np.mean(aps)) * 100, 2)


def load_head(path, key="img_h"):
    h = torch.load(path, map_location="cpu")
    bits = [int(b) for b in h["bits"]]
    bi = bits.index(BIT) if BIT in bits else len(bits) - 1
    layer = NestedHashLayer(h["embed"], h["hidden"], bits, 0.0)
    layer.load_state_dict(h[key])
    layer.eval()
    return layer, bi, h


@torch.no_grad()
def codes(layer, bi, emb):
    return layer(emb)[bi]["binary"].float()


def e5_text_codes():
    """e5-small EN caption emb (mean-pool, L2) -> txt_h_e5 1-bit codes."""
    from transformers import AutoModel, AutoTokenizer
    ck = torch.load(E5_HEAD, map_location="cpu")
    student, prefix = ck["student"], (ck.get("prefix") or "")
    bits = [int(b) for b in ck["bits"]]
    bi = bits.index(BIT) if BIT in bits else len(bits) - 1
    th = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0)
    th.load_state_dict(ck["txt_h"]); th.eval()
    m = AutoModel.from_pretrained(student).eval()
    tok = AutoTokenizer.from_pretrained(student)
    out = []
    for s in range(0, len(en_caps), 256):
        txt = [prefix + x for x in en_caps[s:s + 256]]
        t = tok(txt, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        with torch.no_grad():
            o = m(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state
        msk = t["attention_mask"].unsqueeze(-1).float()
        e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        out.append(F.normalize(e, dim=1).float())
    e5emb = torch.cat(out)
    with torch.no_grad():
        return th(e5emb)[bi]["binary"].float()


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    # ft113 image gallery + server text codes (norm_in=1)
    img_h, ibi, hk = load_head(HEAD_PATH, "img_h")
    txt_h, tbi, _ = load_head(HEAD_PATH, "txt_h")
    norm_in = int(hk.get("norm_in", 1))
    xi = ti_n if norm_in else ti_raw
    xt = tt_n if norm_in else tt_raw
    ic = codes(img_h, ibi, xi)            # ft113 image codes (shared gallery)
    tc_server = codes(txt_h, tbi, xt)     # so400m -> ft113 txt_h
    tc_e5 = e5_text_codes()               # e5 -> txt_h_e5

    rows = []

    def emit(row, task, scores, smaller, exclude_self=False):
        r = {"row": row, "task": task,
             "mAP10": mapk(scores, smaller, 10, exclude_self),
             "mAP_R": mapk(scores, smaller, 0, exclude_self),
             "mAP100": mapk(scores, smaller, 100, exclude_self)}
        rows.append(r)
        print(f"[catmap] {row:28} {task}: mAP@10 {r['mAP10']:6} mAP@R {r['mAP_R']:6} mAP@100 {r['mAP100']:6}", flush=True)

    # float ceiling (cosine)
    emit("float (ceiling)", "T2I", tt_n @ ti_n.t(), smaller=False)
    emit("float (ceiling)", "I2I", ti_n @ ti_n.t(), smaller=False, exclude_self=True)
    # our deployed 1-bit codes
    emit("server (so400m+ft113)", "T2I", hamming(tc_server, ic), smaller=True)
    emit("deployed offline (e5)", "T2I", hamming(tc_e5, ic), smaller=True)
    emit("ft113 image codes", "I2I", hamming(ic, ic), smaller=True, exclude_self=True)

    with open(PAPER / "coco_category_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "task", "mAP10", "mAP_R", "mAP100"])
        w.writeheader(); w.writerows(rows)
    print("[catmap] RESULT_JSON " + json.dumps({"bit": BIT, "C": C, "avg_rel": round(avg_rel, 1),
          "rows": rows}, ensure_ascii=False), flush=True)
    print(f"[catmap] DONE -> paper/coco_category_map.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
