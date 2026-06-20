"""(e) [AUXILIARY] CMH-literature compatibility table on MIRFLICKR-25K — SpikeHash-style.

*** DIFFERENT TASK from our main paper. *** Label-based category retrieval (relevance = share >=1
of 24 MIRFLICKR concepts), NOT instance retrieval. We use frozen SigLIP2-So400m purely as a feature
extractor (vision tower for images; text tower on the user-tag string for the text modality) and train
ONLY a small label-supervised hash head (DCMH-style pairwise loss) at 16/32/64 bit. NO backbone training.
Cited DCMH/SSAH/PromptHash/SpikeHash rows are quoted from their papers (not reproduced here).

Protocol (DCMH-standard MIRFLICKR-25K): keep images with >=1 of 24 labels; query=2000, train=10000
(sampled from the rest), database=remaining. Cross-modal category-mAP@all (I->T and T->I), relevance =
share >=1 label. Frozen-feature codes; sign at eval.

Run on DGX (after core a-d):
  MIR_ROOT=~/data/mirflickr .venv/bin/python web/paper_cmh_benchmark.py [--bits 16,32,64]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import MODEL as SIGLIP_MODEL  # noqa: E402

PAPER = Path(REPO) / "paper"
MIR = Path(os.path.expanduser(os.environ.get("MIR_ROOT", "~/data/mirflickr")))
FEAT_CACHE = os.environ.get("MIR_FEATS", "/tmp/mir_feats.pt")
LABELS = ["sky", "clouds", "water", "sea", "river", "lake", "people", "portrait", "male", "female",
          "baby", "night", "plant_life", "tree", "flower", "animals", "dog", "bird", "structures",
          "sunset", "indoor", "transport", "car", "food"]  # 24 base concepts


def load_labels():
    """image_id (1-indexed) -> 24-d multihot, from annotations/{label}.txt."""
    ann = MIR / "annotations"
    mh = {}
    C = len(LABELS)
    for ci, lab in enumerate(LABELS):
        f = ann / f"{lab}.txt"
        if not f.exists():
            print(f"[e] WARN label file missing: {f}", flush=True)
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            iid = int(line)
            mh.setdefault(iid, np.zeros(C, dtype=np.float32))[ci] = 1.0
    return mh, C


def read_tags(iid):
    for d in ("tags", "tags_raw"):
        p = MIR / "mirflickr" / "meta" / d / f"tags{iid}.txt"
        if p.exists():
            return " ".join(t.strip() for t in open(p, encoding="utf-8", errors="ignore") if t.strip())
    return ""


@torch.no_grad()
def encode_features(ids, dev):
    """frozen SigLIP2: image (vision tower) + text (text tower on tag string) -> 1152-d each."""
    from transformers import AutoModel
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    except Exception:
        from transformers import GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
        proc = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(SIGLIP_MODEL),
                               tokenizer=GemmaTokenizer.from_pretrained(SIGLIP_MODEL))
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer

    def _pool(o):
        return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

    img, txt, t0 = [], [], time.perf_counter()
    B = 64
    for s in range(0, len(ids), B):
        chunk = ids[s:s + B]
        imgs = [Image.open(MIR / "mirflickr" / f"im{i}.jpg").convert("RGB") for i in chunk]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, torch.bfloat16)
        img.append(_pool(bb.vision_model(pixel_values=px)).float().cpu())
        tags = [read_tags(i) or "photo" for i in chunk]
        t = tok(tags, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        txt.append(_pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                   attention_mask=am.to(dev) if am is not None else None)).float().cpu())
        if (s // B) % 20 == 0:
            print(f"  [e] feat {min(s+B,len(ids))}/{len(ids)} ({(s+len(chunk))/(time.perf_counter()-t0+1e-9):.0f}/s)", flush=True)
    return torch.cat(img), torch.cat(txt)


class HashHead(nn.Module):
    def __init__(self, d, bits, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, bits))

    def forward(self, x):
        return torch.tanh(self.net(x))


def train_dcmh(img_f, txt_f, S_full, tr_idx, bits, dev, epochs=150, bs=256, gamma=1.0, eta=1.0, seed=42):
    """DCMH-style cross-modal pairwise hashing on frozen features (minibatch label-similarity)."""
    torch.manual_seed(seed)
    d = img_f.shape[1]
    ih, th = HashHead(d, bits).to(dev), HashHead(d, bits).to(dev)
    opt = torch.optim.Adam(list(ih.parameters()) + list(th.parameters()), lr=1e-3)
    I = img_f[tr_idx].to(dev)
    T = txt_f[tr_idx].to(dev)
    S = S_full[tr_idx][:, tr_idx].to(dev)  # (ntr, ntr) in {0,1}
    n = I.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            F_ = ih(I[idx]); G_ = th(T[idx])
            Sb = S[idx][:, idx]
            theta = 0.5 * (F_ @ G_.t())
            # pairwise neg-log-likelihood: log(1+e^theta) - S*theta
            pair = (torch.log1p(torch.exp(-torch.abs(theta))) + torch.clamp(theta, min=0) - Sb * theta).mean()
            quant = ((torch.sign(F_) - F_) ** 2).mean() + ((torch.sign(G_) - G_) ** 2).mean()
            balance = (F_.mean(0) ** 2).mean() + (G_.mean(0) ** 2).mean()
            loss = pair + gamma * quant + eta * balance
            opt.zero_grad(); loss.backward(); opt.step()
    ih.eval(); th.eval()
    return ih, th


@torch.no_grad()
def codes(head, feat, idx, dev):
    return torch.sign(head(feat[idx].to(dev))).cpu()


def mAP(query_codes, db_codes, query_mh, db_mh, kcap=0):
    """cross-modal category mAP@all (or @kcap). relevance = share >=1 label. Hamming ranking."""
    rel = (query_mh @ db_mh.t() > 0)  # (Nq, Ndb) bool
    sim = query_codes @ db_codes.t()  # higher = closer (codes ±1)
    order = sim.argsort(dim=1, descending=True)
    Nq = query_codes.shape[0]
    aps = []
    K = kcap if kcap > 0 else db_codes.shape[0]
    ar = torch.arange(1, K + 1, dtype=torch.float32)
    for i in range(Nq):
        r = rel[i, order[i, :K]].float()
        nr = r.sum()
        if nr == 0:
            continue
        prec = torch.cumsum(r, 0) / ar
        aps.append(((prec * r).sum() / nr).item())
    return round(100 * float(np.mean(aps)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="16,32,64")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PAPER.mkdir(parents=True, exist_ok=True)
    bits_list = [int(b) for b in args.bits.split(",")]

    mh_map, C = load_labels()
    all_imgs = sorted(int(Path(p).stem[2:]) for p in glob.glob(str(MIR / "mirflickr" / "im*.jpg")))
    ids = [i for i in all_imgs if i in mh_map]  # images with >=1 label
    print(f"[e] MIRFLICKR images total={len(all_imgs)} with>=1label={len(ids)} labels={C}", flush=True)
    mh = torch.from_numpy(np.stack([mh_map[i] for i in ids]))  # (N, 24)

    if os.path.exists(FEAT_CACHE):
        ck = torch.load(FEAT_CACHE, map_location="cpu")
        if ck.get("ids") == ids:
            img_f, txt_f = ck["img"], ck["txt"]
            print(f"[e] loaded cached features {tuple(img_f.shape)}", flush=True)
        else:
            img_f = txt_f = None
    else:
        img_f = txt_f = None
    if img_f is None:
        img_f, txt_f = encode_features(ids, dev)
        torch.save({"ids": ids, "img": img_f, "txt": txt_f}, FEAT_CACHE)
        print(f"[e] cached features -> {FEAT_CACHE}", flush=True)

    # SpikeHash protocol (so our row is directly comparable to its table): query=2000, train=5000,
    # database=remaining (~18015). mAP@50 primary; we also report mAP@all (DCMH/SSAH protocol).
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(ids), generator=g)
    q_idx = perm[:2000]
    db_idx = perm[2000:]              # retrieval database
    tr_idx = db_idx[:5000]           # SpikeHash uses 5000 training samples
    S_full = (mh @ mh.t() > 0).float()
    print(f"[e] split query={len(q_idx)} database={len(db_idx)} train={len(tr_idx)} labels={C}", flush=True)

    def row(method, bits, i2t50, t2i50, i2ta, t2ia, protocol, source):
        return {"method": method, "dataset": "MIRFLICKR-25K", "bits": bits, "mAP_I2T": i2t50,
                "mAP_T2I": t2i50, "mAP_I2T_all": i2ta, "mAP_T2I_all": t2ia, "protocol": protocol, "source": source}

    rows = []
    for bits in bits_list:
        t0 = time.perf_counter()
        ih, th = train_dcmh(img_f, txt_f, S_full, tr_idx, bits, dev, epochs=args.epochs)
        qI = codes(ih, img_f, q_idx, dev); qT = codes(th, txt_f, q_idx, dev)
        dbI = codes(ih, img_f, db_idx, dev); dbT = codes(th, txt_f, db_idx, dev)
        qmh, dbmh = mh[q_idx], mh[db_idx]
        i2t50 = mAP(qI, dbT, qmh, dbmh, kcap=50); t2i50 = mAP(qT, dbI, qmh, dbmh, kcap=50)
        i2ta = mAP(qI, dbT, qmh, dbmh, kcap=0); t2ia = mAP(qT, dbI, qmh, dbmh, kcap=0)
        rows.append(row("SigLIP2-feat + label head (ours, frozen backbone)", bits, i2t50, t2i50, i2ta, t2ia,
                        "q2000/tr5000/db~18015/24lab", "this work (measured)"))
        print(f"[e] bits {bits}: mAP@50 I2T {i2t50}/T2I {t2i50} | mAP@all I2T {i2ta}/T2I {t2ia} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- cited baselines (quoted from papers; NOT reproduced). mAP shown in its paper's protocol column. ----
    SRC_SH = "SpikeHash arXiv:2606.00740 Tab. (mAP@50, CLIP-feature methods)"
    SRC_SSAH = "SSAH arXiv:1804.01223 Tab. (mAP@all, AlexNet+BoW, end-to-end)"
    cited = [
        ("UCMFH", {16: (91.8, 92.1), 32: (95.0, 94.8), 64: (96.0, 96.0)}, "mAP@50", SRC_SH),
        ("DDSS", {16: (94.7, 94.8), 32: (96.3, 96.5), 64: (96.9, 96.8)}, "mAP@50", SRC_SH),
        ("SpikeHash", {16: (93.2, 93.3), 32: (95.1, 95.0), 64: (95.8, 95.8)}, "mAP@50", SRC_SH),
        ("DCMH", {16: (74.1, 74.1), 32: (74.7, 74.7), 64: (74.9, 74.9)}, "mAP@all", SRC_SSAH),
        ("SSAH", {16: (77.9, 78.2), 32: (79.1, 79.0), 64: (79.9, 80.0)}, "mAP@all", SRC_SSAH),
    ]
    for name, perbit, proto, src in cited:
        for bits, (i2t, t2i) in perbit.items():
            if proto == "mAP@50":
                rows.append(row(name, bits, i2t, t2i, "", "", proto, src))
            else:
                rows.append(row(name, bits, "", "", i2t, t2i, proto, src))

    with open(PAPER / "cmh_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "dataset", "bits", "mAP_I2T", "mAP_T2I",
                                          "mAP_I2T_all", "mAP_T2I_all", "protocol", "source"])
        w.writeheader(); w.writerows(rows)
    print("[e] RESULT_JSON " + json.dumps({"ours": rows}, ensure_ascii=False), flush=True)
    print(f"[e] DONE -> paper/cmh_benchmark.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
