"""(iv) Fair NLLB vs ft113 re-compare with the so400m lowercase fix.

Batch #2 (f) compared NLLB+head (1bit) to SigLIP2+ft113 (1bit) using ORIG-CASE so400m text — so the ft113
column was suppressed on cased langs (e.g. de 30 was the casing-bug value). Here ft113's so400m text is
LOWERCASED (the fix); NLLB+head stays as-is (its own tokenizer, batch #2 numbers). Per-language R@10/mAP10.

Reuses /tmp/xm_so400m.pt (XM3600 so400m image gallery + cached ORIG-case per-lang text emb, built by (g)) +
ft113 head; only the LOWERCASED so400m text is re-encoded. NLLB+head numbers read from paper/nllb_hashing.csv.

Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_nllb_compare_lc.py --data data/xm3600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import MODEL as SIGLIP_MODEL, Encoder, pack_bits  # noqa: E402
from web.paper_baselines_multiling import xm3600_data  # noqa: E402

PAPER = Path(REPO) / "paper"
REPR = ["de", "te", "th", "hi", "en", "ko"]
CACHE = "/tmp/xm_so400m.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def r10_map(ix, q, gold):
    _, I = ix.search(q, 10)
    r = round(100 * np.mean([gold[i] in I[i, :10] for i in range(len(gold))]), 2)
    ap = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0) for i in range(len(gold))]
    return r, round(100 * float(np.mean(ap)), 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/xm3600"); args = ap.parse_args()
    PAPER.mkdir(parents=True, exist_ok=True)
    ds = xm3600_data(args.data)
    cache = torch.load(CACHE, map_location="cpu")  # img_emb + per_lang (orig-case text_emb + gold)
    enc = Encoder()
    from transformers import AutoModel, GemmaTokenizer
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    tok = GemmaTokenizer.from_pretrained(SIGLIP_MODEL)

    @torch.no_grad()
    def so400m_lower(strings):
        out = []
        for s in range(0, len(strings), 256):
            t = tok([x.lower() for x in strings[s:s + 256]], padding="max_length", max_length=64,
                    truncation=True, return_tensors="pt")
            am = t.get("attention_mask")
            o = bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)
            e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
            out.append(e.float().cpu())
        return torch.cat(out)

    gal = enc.image_codes_packed(cache["img_emb"].numpy()); ix = faiss_bin(gal, enc.bits)

    # NLLB+head numbers (batch #2)
    nllb = {}
    nf = PAPER / "nllb_hashing.csv"
    if nf.exists():
        for row in csv.DictReader(open(nf)):
            if row["model"] == "NLLB+head" and row["dataset"] == "xm3600":
                nllb[row["lang"]] = (float(row["R10"]), float(row["mAP10"]))

    rows = []
    orig_r, low_r = [], []
    for L in ds["langs"]:
        caps, gold = ds["caps"][L], ds["gold"][L]
        te_orig = cache["per_lang"][L]["text_emb"]                 # cached orig-case
        co_r, co_m = r10_map(ix, pack_bits(enc._codes_pm1(enc.txt_h, te_orig)), gold)
        te_low = so400m_lower(caps)                                # re-encoded lowercased
        cl_r, cl_m = r10_map(ix, pack_bits(enc._codes_pm1(enc.txt_h, te_low)), gold)
        orig_r.append(co_r); low_r.append(cl_r)
        if L in REPR:
            nr, nm = nllb.get(L, ("", ""))
            rows.append({"lang": L, "ft113_orig_R10": co_r, "ft113_lower_R10": cl_r, "ft113_lower_mAP10": cl_m,
                         "nllb_head_R10": nr, "nllb_head_mAP10": nm})
            print(f"[iv] {L}: ft113 orig {co_r} -> lower {cl_r} | NLLB+head {nr}", flush=True)
    rows.append({"lang": "avg36", "ft113_orig_R10": round(float(np.mean(orig_r)), 2),
                 "ft113_lower_R10": round(float(np.mean(low_r)), 2), "ft113_lower_mAP10": "",
                 "nllb_head_R10": nllb.get("avg36", ("", ""))[0], "nllb_head_mAP10": nllb.get("avg36", ("", ""))[1]})
    print(f"[iv] avg36: ft113 orig {rows[-1]['ft113_orig_R10']} -> lower {rows[-1]['ft113_lower_R10']} | "
          f"NLLB+head {rows[-1]['nllb_head_R10']}", flush=True)

    with open(PAPER / "nllb_compare_lc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "ft113_orig_R10", "ft113_lower_R10", "ft113_lower_mAP10",
                                          "nllb_head_R10", "nllb_head_mAP10"])
        w.writeheader(); w.writerows(rows)
    print("[iv] RESULT_JSON " + json.dumps({"rows": rows}, ensure_ascii=False), flush=True)
    print(f"[iv] DONE -> paper/nllb_compare_lc.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
