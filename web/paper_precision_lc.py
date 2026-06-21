"""(1-prec) Precision table (tab:prec) lowercased re-derive + case-invariance confirmation.

precision.csv (orig-case) quantizes three targets — the hash HEAD, the stored EMBEDDING, and the so400m
TEXT TOWER — to fp16/bf16/int8 and reports: bitflips_per_1024 (vs fp32 codes), top10_overlap (retrieval
set overlap vs fp32), EN_R10, KO_R10. There is no reusable generator script (it was ad-hoc), so this
reconstructs the experiment and (a) validates the orig-case numbers against the recorded anchors
[fp16 head ~0.12 / bf16 head ~0.95 / int8 head ~18.97 flips; fp32 EN_R10 ~79.9], then (b) re-runs with
so400m text LOWERCASED. Hypothesis (work order): bitflips/overlap are CODE STATISTICS -> case-invariant;
only the absolute EN_R10 shifts ~+1.3. KO is caseless -> unchanged.

int8 scheme (fixed, applied identically to orig & lower so the delta is the finding): head = per-output-
channel symmetric int8 of each nn.Linear weight; emb = per-row symmetric int8 of the embedding vector.

Reuses caches from paper_coco_lc_full.py (no backbone reload): /tmp/coco_en_{orig,lc}.pt (bf16->float),
/tmp/coco_en_{orig,lc}_fp32.pt (fp32 backbone, for the text_tower bf16-vs-fp32 row). KO from coco_ko_test.

Run: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_precision_lc.py
"""
from __future__ import annotations

import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def rk(ix, q_packed, gold):
    _, I = ix.search(np.ascontiguousarray(q_packed), 10)
    out = {k: round(100 * float(np.mean([gold[i] in I[i, :k] for i in range(len(gold))])), 2) for k in KS}
    ap = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0)
          for i in range(len(gold))]
    out["map10"] = round(100 * float(np.mean(ap)), 2)
    out["_I"] = I[:, :10]
    return out


# int8 scheme: the recorded precision.csv int8 values (head 18.97 / emb 5.45 flips) are coarse, consistent
# with PER-TENSOR symmetric int8 (one scale per tensor) rather than fine per-channel/per-row. Default to
# per-tensor to match; INT8_MODE=channel selects the gentler per-channel/per-row scheme. fp16/bf16/text_tower
# are scheme-free and reproduce the recorded anchors exactly either way.
INT8_MODE = os.environ.get("INT8_MODE", "tensor")


def int8_head(W):
    scale = (W.abs().amax(dim=1, keepdim=True) if INT8_MODE == "channel" else W.abs().max()).clamp(min=1e-12) / 127.0
    return (W / scale).round().clamp(-127, 127) * scale


def int8_emb(X):
    scale = (X.abs().amax(dim=1, keepdim=True) if INT8_MODE == "channel" else X.abs().max()).clamp(min=1e-12) / 127.0
    return (X / scale).round().clamp(-127, 127) * scale


def quant_head(head, dtype):
    h = copy.deepcopy(head)
    if dtype == "int8":
        with torch.no_grad():
            for m in h.modules():
                if isinstance(m, nn.Linear):
                    m.weight.copy_(int8_head(m.weight.data))
        return h.to(dev).float().eval()
    return h.to(dev).to({"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]).eval()


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    bidx = enc.bit_index

    en_orig = torch.load("/tmp/coco_en_orig.pt", map_location="cpu")
    en_low = torch.load("/tmp/coco_en_lc.pt", map_location="cpu")
    en_orig32 = torch.load("/tmp/coco_en_orig_fp32.pt", map_location="cpu") if os.path.exists("/tmp/coco_en_orig_fp32.pt") else None
    en_low32 = torch.load("/tmp/coco_en_lc_fp32.pt", map_location="cpu") if os.path.exists("/tmp/coco_en_lc_fp32.pt") else None
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    ko_emb = KO["txt_emb"].float()
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    te_img = EC["test"]["img"].float().numpy()
    n = en_orig.shape[0]; gold = list(range(n))

    gal = enc.image_codes_packed(te_img); ix = faiss_bin(gal, enc.bits)

    @torch.no_grad()
    def codes_fp32(emb):  # reference fp32 head codes (±1)
        return enc._codes_pm1(enc.txt_h, emb)

    @torch.no_grad()
    def codes_head(hq, emb, dtype):
        inp = enc._prep(emb.to(dev))
        if dtype in ("fp16", "bf16"):
            inp = inp.to({"fp16": torch.float16, "bf16": torch.bfloat16}[dtype])
        return hq(inp)[bidx]["binary"].float().cpu().numpy()

    rows = []

    def measure(target, dtype, variant, emb, emb_ref_fp32=None):
        """Return dict row: bitflips_per_1024, top10_overlap, R@{1,5,10}, mAP10 for one (target,dtype,variant)."""
        ref_codes = codes_fp32(emb)               # fp32-head, fp32-emb baseline for THIS emb
        ref = rk(ix, pack_bits(ref_codes), gold)
        if target == "fp32":
            q_codes = ref_codes
        elif target == "head":
            q_codes = codes_head(quant_head(enc.txt_h, dtype), emb, dtype)
        elif target == "emb":
            cast = {"fp16": torch.float16, "bf16": torch.bfloat16}
            if dtype == "int8":
                emb_q = int8_emb(emb)
            else:
                emb_q = emb.to(cast[dtype]).float()
            q_codes = codes_fp32(emb_q)
        elif target == "text_tower":   # dtype bf16: emb already bf16-backbone; ref is fp32-backbone
            assert emb_ref_fp32 is not None
            ref_codes = codes_fp32(emb_ref_fp32); ref = rk(ix, pack_bits(ref_codes), gold)
            q_codes = codes_fp32(emb)
        else:
            raise ValueError(target)
        q = rk(ix, pack_bits(q_codes), gold)
        bitflips = float((q_codes != ref_codes).sum(1).mean())
        overlap = float(np.mean([len(set(q["_I"][i]) & set(ref["_I"][i])) / 10.0 for i in range(n)]))
        return {"target": target, "dtype": dtype, "variant": variant,
                "bitflips_per_1024": round(bitflips, 2), "top10_overlap": round(overlap, 4),
                "EN_R1": q[1], "EN_R5": q[5], "EN_R10": q[10], "EN_mAP10": q["map10"]}

    plan = [("fp32", "fp32"), ("head", "fp16"), ("head", "bf16"), ("head", "int8"),
            ("emb", "fp16"), ("emb", "bf16"), ("emb", "int8")]
    for variant, emb, emb32 in (("orig", en_orig, en_orig32), ("lower", en_low, en_low32)):
        for target, dtype in plan:
            rows.append(measure(target, dtype, variant, emb))
        if emb32 is not None:
            rows.append(measure("text_tower", "bf16", variant, emb, emb_ref_fp32=emb32))
        else:
            rows.append({"target": "text_tower", "dtype": "bf16", "variant": variant,
                         "bitflips_per_1024": "", "top10_overlap": "", "EN_R1": "", "EN_R5": "",
                         "EN_R10": "FLAGGED(no fp32 cache)", "EN_mAP10": ""})

    # KO (caseless): fp32 baseline + int8 head, to confirm KO unchanged
    ko_rows = []
    ref_ko = rk(ix, pack_bits(codes_fp32(ko_emb)), gold)
    ko_rows.append({"target": "fp32", "dtype": "fp32", "lang": "KO", "EN_R10": "", "KO_R10": ref_ko[10]})
    ko_q = codes_head(quant_head(enc.txt_h, "int8"), ko_emb, "int8")
    ko_rows.append({"target": "head", "dtype": "int8", "lang": "KO", "EN_R10": "",
                    "KO_R10": rk(ix, pack_bits(ko_q), gold)[10]})

    with open(PAPER / "precision_lc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "dtype", "variant", "bitflips_per_1024", "top10_overlap",
                                          "EN_R1", "EN_R5", "EN_R10", "EN_mAP10"])
        w.writeheader(); w.writerows(rows)

    # validation + invariance summary
    def get(target, dtype, variant, key):
        for r in rows:
            if r["target"] == target and r["dtype"] == dtype and r["variant"] == variant:
                return r[key]
        return None
    print("[prec] VALIDATE orig vs recorded: fp32 EN_R10", get("fp32", "fp32", "orig", "EN_R10"),
          "(rec 79.92) | head int8 flips", get("head", "int8", "orig", "bitflips_per_1024"), "(rec 18.97) |",
          "head bf16 flips", get("head", "bf16", "orig", "bitflips_per_1024"), "(rec 0.95) |",
          "head fp16 flips", get("head", "fp16", "orig", "bitflips_per_1024"), "(rec 0.12)", flush=True)
    for target, dtype in plan[1:] + [("text_tower", "bf16")]:
        bo, bl = get(target, dtype, "orig", "bitflips_per_1024"), get(target, dtype, "lower", "bitflips_per_1024")
        ro, rl = get(target, dtype, "orig", "EN_R10"), get(target, dtype, "lower", "EN_R10")
        print(f"[prec] {target}/{dtype}: flips orig {bo} ~ lower {bl} (case-inv?) | EN_R10 {ro} -> {rl}", flush=True)
    print("[prec] KO:", ko_rows, flush=True)
    print("[prec] RESULT_JSON " + json.dumps({"rows": rows, "ko": ko_rows}, ensure_ascii=False), flush=True)
    print(f"[prec] DONE -> paper/precision_lc.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
