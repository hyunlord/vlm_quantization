"""AAAI rigor §3 (Pillar 2) — cross-modal binary-parity failure, matched comparison.

Three text paths to the SAME frozen ft113 code space (gallery = img_h(test_img) codes), COCO 5K test, I2T:
  - CEILING:      true SigLIP2-text → txt_h → bits.
  - DISTILLATION: e5 (fine-tuned backbone + proj, distill_e5.pt) → predicted SigLIP2-text → txt_h → bits.
  - HEAD-ADAPT:   e5 (frozen base) → txt_h' (txt_h_e5.pt, InfoNCE-trained to img codes) → bits.
Report together: cosine(pred, true SigLIP-text) [distill only], continuous R@10 (cosine pred↔img),
binary-parity (% of 1024 bits equal to the true SigLIP-text code), and binary R@10 — to characterize
"cosine high but binary R@K collapses" and contrast with head-adapt (optimizes codes directly).

Run on DGX:
  .venv/bin/python web/rigor_parity.py --out paper/rigor_parity.csv
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
TEST_CACHE = os.environ.get("TEST_CACHE", "/tmp/emb_cache.pt")
DISTILL = os.environ.get("DISTILL", "/tmp/distill_e5.pt")
TXTH_E5 = os.environ.get("TXTH_E5", "/tmp/txt_h_e5.pt")
E5_TEST_EN = os.environ.get("E5_TEST_EN", "/tmp/e5_test_en.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def nested(ckpath_or_state, embed, hidden, bits, key=None):
    h = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval()
    st = torch.load(ckpath_or_state, map_location="cpu") if isinstance(ckpath_or_state, str) else ckpath_or_state
    if key:
        st = st[key]
    h.load_state_dict(st)
    return h


def head_bits(head, x, B=1024):
    raw = head.hash_head(x)
    z = F.normalize(head.batch_norms[-1](raw[:, :B]), p=2, dim=1)
    return (z > 0).cpu().numpy()


def pack(b01):
    return np.ascontiguousarray(np.packbits(b01.astype(np.uint8), axis=1, bitorder="big"))


def bin_r10(qb, gb):
    import faiss
    ix = faiss.IndexBinaryFlat(qb.shape[1]); ix.add(pack(gb))
    _, I = ix.search(pack(qb), 10)
    return round(100 * np.mean([r in I[r] for r in range(qb.shape[0])]), 2)


def cont_r10(q, g):
    sims = q @ g.T
    order = np.argsort(-sims, axis=1)[:, :10]
    return round(100 * np.mean([r in order[r] for r in range(q.shape[0])]), 2)


def run_distill(captions):
    from transformers import AutoModel, AutoTokenizer
    ck = torch.load(DISTILL, map_location="cpu")
    student = ck["student"]; maxlen = int(ck["maxlen"])
    m = AutoModel.from_pretrained(student)
    m.load_state_dict(ck["backbone"]); m = m.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(student)
    proj = nn.Linear(ck["proj"]["weight"].shape[1], ck["proj"]["weight"].shape[0])
    proj.load_state_dict(ck["proj"]); proj = proj.to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(captions), 256):
            t = tok(captions[s:s + 256], padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
            o = m(t["input_ids"].to(dev), t["attention_mask"].to(dev)).last_hidden_state
            msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
            e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            out.append(F.normalize(proj(e), dim=1).cpu())
    return torch.cat(out, 0)  # (N,1152) predicted SigLIP-text, L2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper/rigor_parity.csv")
    args = ap.parse_args()
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]; embed, hidden = ck["embed"], ck["hidden"]
    img_h = nested(ck["img_h"], embed, hidden, bits)
    txt_h = nested(ck["txt_h"], embed, hidden, bits)

    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    img = F.normalize(EC["img"].float(), dim=1).to(dev)
    true_txt = F.normalize(EC["txt"].float(), dim=1).to(dev)
    caps = [str(c) for c in EC["captions"]]
    img_np = img.cpu().numpy()

    img_bits = head_bits(img_h, img)
    true_bits = head_bits(txt_h, true_txt)
    rows = []

    # CEILING
    rows.append({"path": "ceiling(true SigLIP-text)", "cosine_to_true": 1.0, "binary_parity_pct": 100.0,
                 "cont_R10": cont_r10(true_txt.cpu().numpy(), img_np), "bin_R10": bin_r10(true_bits, img_bits)})

    # DISTILLATION
    pred = run_distill(caps).to(dev)
    cos = float(F.cosine_similarity(pred, true_txt, dim=1).mean())
    pred_bits = head_bits(txt_h, pred)
    parity = round(100 * float((pred_bits == true_bits).mean()), 2)
    rows.append({"path": "distillation(e5→SigLIP-text)", "cosine_to_true": round(cos, 4), "binary_parity_pct": parity,
                 "cont_R10": cont_r10(pred.cpu().numpy(), img_np), "bin_R10": bin_r10(pred_bits, img_bits)})

    # HEAD-ADAPT
    e5 = torch.load(E5_TEST_EN, map_location="cpu").float().to(dev)
    # txt_h_e5.pt may be a raw state_dict or wrapped; load robustly
    st = torch.load(TXTH_E5, map_location="cpu")
    st = st.get("txt_h", st) if isinstance(st, dict) and "txt_h" in st else st
    ha = NestedHashLayer(e5.shape[1], hidden, bits, 0.0).to(dev).eval(); ha.load_state_dict(st)
    ha_bits = head_bits(ha, F.normalize(e5, dim=1))
    parity_ha = round(100 * float((ha_bits == true_bits).mean()), 2)
    rows.append({"path": "head-adapt(e5→txt_h')", "cosine_to_true": float("nan"), "binary_parity_pct": parity_ha,
                 "cont_R10": float("nan"), "bin_R10": bin_r10(ha_bits, img_bits)})

    for r in rows:
        print(f"[parity] {r['path']:>28}: cos {r['cosine_to_true']} parity {r['binary_parity_pct']}% "
              f"cont_R10 {r['cont_R10']} bin_R10 {r['bin_R10']}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "cosine_to_true", "binary_parity_pct", "cont_R10", "bin_R10"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
