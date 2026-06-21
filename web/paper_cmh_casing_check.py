"""(4) CMH (MIRFLICKR-25K) casing-independence check — low-cost confirmation.

The CMH table (e) uses frozen SigLIP2 IMAGE features + the so400m text tower on the user-TAG string (not
English captions). Hypothesis: lowercasing tags barely moves the features (tags are largely already lowercase
keywords), so category-mAP is casing-independent and cmh_benchmark.csv stands as-is.

Evidence (no full re-run): (1) fraction of tag strings changed by .lower(); (2) mean cos(orig tag feature,
lowercased tag feature) — reuses cached image features, only re-encodes the tag text; (3) ONE bit (64) DCMH
head trained on orig vs lowercased tag features -> category-mAP@50/@all delta (should be ~noise).

Run: MIR_ROOT=~/data/mirflickr MIR_FEATS=/tmp/mir_feats.pt \
     .venv/bin/python web/paper_cmh_casing_check.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import MODEL as SIGLIP_MODEL  # noqa: E402
from web.paper_cmh_benchmark import load_labels, read_tags, train_dcmh, codes, mAP  # noqa: E402

PAPER = Path(REPO) / "paper"
FEAT_CACHE = os.environ.get("MIR_FEATS", "/tmp/mir_feats.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _pool(o):
    return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)


@torch.no_grad()
def encode_tags(ids, lower):
    from transformers import AutoModel
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(SIGLIP_MODEL).tokenizer
    except Exception:
        from transformers import GemmaTokenizer
        tok = GemmaTokenizer.from_pretrained(SIGLIP_MODEL)
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    out, B = [], 64
    tags_all = [read_tags(i) or "photo" for i in ids]
    changed = sum(1 for t in tags_all if t != t.lower())
    for s in range(0, len(ids), B):
        chunk = tags_all[s:s + B]
        if lower:
            chunk = [t.lower() for t in chunk]
        t = tok(chunk, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        out.append(_pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                  attention_mask=am.to(dev) if am is not None else None)).float().cpu())
    return torch.cat(out), changed, len(tags_all)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    ck = torch.load(FEAT_CACHE, map_location="cpu")
    ids, img_f, txt_orig = ck["ids"], ck["img"], ck["txt"]
    mh_map, C = load_labels()
    mh = torch.from_numpy(np.stack([mh_map[i] for i in ids]))

    # (1)+(2) tag-feature drift under lowercasing
    txt_low, changed, ntot = encode_tags(ids, lower=True)
    cos = float(F.cosine_similarity(F.normalize(txt_orig, dim=1), F.normalize(txt_low, dim=1)).mean())
    frac_changed = round(100 * changed / max(ntot, 1), 2)
    print(f"[e-case] tags changed by .lower(): {changed}/{ntot} ({frac_changed}%) | "
          f"mean cos(orig tag feat, lower tag feat) = {cos:.4f}", flush=True)

    # (3) ONE bit (64) DCMH on orig vs lower tag features -> category-mAP delta
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(ids), generator=g)
    q_idx, db_idx = perm[:2000], perm[2000:]
    tr_idx = db_idx[:5000]
    S_full = (mh @ mh.t() > 0).float()
    qmh, dbmh = mh[q_idx], mh[db_idx]

    def run(txt_f, tag):
        t0 = time.perf_counter()
        ih, th = train_dcmh(img_f, txt_f, S_full, tr_idx, 64, dev, epochs=150)
        qI, qT = codes(ih, img_f, q_idx, dev), codes(th, txt_f, q_idx, dev)
        dbI, dbT = codes(ih, img_f, db_idx, dev), codes(th, txt_f, db_idx, dev)
        r = {"tags": tag, "bits": 64,
             "mAP50_I2T": mAP(qI, dbT, qmh, dbmh, kcap=50), "mAP50_T2I": mAP(qT, dbI, qmh, dbmh, kcap=50),
             "mAPall_I2T": mAP(qI, dbT, qmh, dbmh, kcap=0), "mAPall_T2I": mAP(qT, dbI, qmh, dbmh, kcap=0)}
        print(f"[e-case] {tag}: {r} ({time.perf_counter()-t0:.0f}s)", flush=True)
        return r

    r_orig = run(txt_orig, "orig")
    r_low = run(txt_low, "lower")
    deltas = {k: round(r_low[k] - r_orig[k], 2) for k in ("mAP50_I2T", "mAP50_T2I", "mAPall_I2T", "mAPall_T2I")}
    casing_independent = max(abs(v) for v in deltas.values()) < 1.0

    with open(PAPER / "cmh_casing_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tags", "bits", "mAP50_I2T", "mAP50_T2I", "mAPall_I2T", "mAPall_T2I"])
        w.writeheader(); w.writerows([r_orig, r_low])
    print(f"[e-case] mAP deltas (lower-orig) @64bit: {deltas} | casing_independent={casing_independent} "
          f"(|delta|<1.0)", flush=True)
    print("[e-case] RESULT_JSON " + json.dumps({"tag_feat_cos": round(cos, 4), "frac_tags_changed": frac_changed,
          "mAP_orig": r_orig, "mAP_lower": r_low, "deltas": deltas,
          "casing_independent": casing_independent}, ensure_ascii=False), flush=True)
    print("[e-case] DONE -> paper/cmh_casing_check.csv", flush=True)


if __name__ == "__main__":
    main()
