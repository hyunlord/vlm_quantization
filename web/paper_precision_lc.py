"""(1-prec) Precision table (tab:prec) lowercased re-derive — FAITHFUL to eval_paper.py's section C.

Reuses the canonical precision scheme from web/eval_paper.py (the script that produced paper/precision.csv):
  head int8 = torch.ao.quantization.quantize_dynamic({Linear}, qint8) on CPU (NOT a weight round-trip);
  emb int8 = per-row storage-cast; flips = pair_bitflip on PACKED codes averaged over EN+KO; top10_overlap
  vs the fp32 image gallery. text_tower = bf16-backbone vs fp32-backbone text encode. This reproduces the
  recorded anchors (head fp16 0.12 / bf16 0.95 / int8 18.97; emb int8 5.45; text_tower 2.57; fp32 EN 79.92).

It runs the whole C section twice — orig-case and lowercased so400m text — to (a) validate the orig column
against precision.csv and (b) confirm the work-order hypothesis: bit-flip / overlap are CODE STATISTICS, so
case-INVARIANT; only EN_R10 shifts ~+1.3. KO is caseless -> unchanged.

Reuses caches from paper_coco_lc_full.py: /tmp/coco_en_lc.pt (lowercased EN, bf16->float),
/tmp/coco_en_{lc,orig}_fp32.pt (fp32 backbone, for text_tower). orig EN = emb_cache (the exact 79.92 cache);
KO = coco_ko_test (caseless). No backbone reload.

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

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import Encoder, pack_bits  # noqa: E402
from web.eval_paper import pair_bitflip, head_codes, faiss_bin, recall_ks, CODE_BITS  # noqa: E402

PAPER = Path(REPO) / "paper"
LANGS = ("EN", "KO")


def cast_emb(x, dt):
    """encoder-output storage-precision cast (faithful to eval_paper.py)."""
    if dt == "int8":
        s = np.abs(x).max(axis=1, keepdims=True) / 127.0 + 1e-9
        return (np.round(x / s).clip(-127, 127) * s).astype(np.float32)
    if dt == "fp16":
        return x.astype(np.float16).astype(np.float32)
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()  # bf16


def run_variant(enc, te_img, TXT, gold, en_bf16=None, en_fp32=None):
    """Replicates eval_paper.py section C for one preprocessing variant; returns list of rows."""
    fp32_img = head_codes(enc.img_h, te_img, enc, CODE_BITS)
    fp32_txt = {L: head_codes(enc.txt_h, TXT[L], enc, CODE_BITS) for L in LANGS}
    ref_img_p = pack_bits(fp32_img)
    ref_txt_p = {L: pack_bits(fp32_txt[L]) for L in LANGS}
    ix_ref = faiss_bin(ref_img_p, CODE_BITS)

    def top10_overlap(qa, qb):
        _, Ia = ix_ref.search(qa, 10); _, Ib = ix_ref.search(qb, 10)
        return float(np.mean([len(set(Ia[i]) & set(Ib[i])) / 10 for i in range(len(qa))]))

    rows = []

    def emit(target, dtype, img_codes, txt_codes, gal_p, note=""):
        ix = faiss_bin(gal_p, CODE_BITS)
        bf, ov, r10 = [], [], {}
        for L in LANGS:
            tp = pack_bits(txt_codes[L])
            bf.append(pair_bitflip(tp, ref_txt_p[L]))
            ov.append(top10_overlap(tp, ref_txt_p[L]))
            r10[L] = recall_ks(ix, tp, gold)[10]
        rows.append({"target": target, "dtype": dtype, "bitflips_per_1024": round(float(np.mean(bf)), 2),
                     "top10_overlap": round(float(np.mean(ov)), 3), "EN_R10": r10["EN"], "KO_R10": r10["KO"],
                     "note": note})

    emit("head", "fp32", fp32_img, fp32_txt, ref_img_p, "baseline")
    # C-head: cast both heads
    import torch.ao.quantization as q
    for dt in ("fp16", "bf16", "int8"):
        if dt == "int8":
            ih = q.quantize_dynamic(copy.deepcopy(enc.img_h).cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
            th = q.quantize_dynamic(copy.deepcopy(enc.txt_h).cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
            ic = head_codes(ih, te_img, enc, CODE_BITS, device="cpu")
            tc = {L: head_codes(th, TXT[L], enc, CODE_BITS, device="cpu") for L in LANGS}
        else:
            d = torch.float16 if dt == "fp16" else torch.bfloat16
            ih = copy.deepcopy(enc.img_h).to(enc.device, d).eval()
            th = copy.deepcopy(enc.txt_h).to(enc.device, d).eval()
            ic = head_codes(ih, te_img, enc, CODE_BITS, in_dtype=d)
            tc = {L: head_codes(th, TXT[L], enc, CODE_BITS, in_dtype=d) for L in LANGS}
        emit("head", dt, ic, tc, pack_bits(ic))
    # C-emb: cast cached embeddings, fp32 head
    for dt in ("fp16", "bf16", "int8"):
        ic = head_codes(enc.img_h, cast_emb(te_img, dt), enc, CODE_BITS)
        tc = {L: head_codes(enc.txt_h, cast_emb(TXT[L], dt), enc, CODE_BITS) for L in LANGS}
        emit("emb", dt, ic, tc, pack_bits(ic), "storage-cast")
    # C-text_tower bf16: so400m tower bf16 vs fp32 backbone encode (EN only; KO caseless ~ flat)
    if en_bf16 is not None and en_fp32 is not None:
        bf16_codes = head_codes(enc.txt_h, en_bf16, enc, CODE_BITS)
        fp32_codes = head_codes(enc.txt_h, en_fp32, enc, CODE_BITS)
        ixg = faiss_bin(ref_img_p, CODE_BITS)
        rows.append({"target": "text_tower", "dtype": "bf16",
                     "bitflips_per_1024": round(pair_bitflip(pack_bits(bf16_codes), pack_bits(fp32_codes)), 2),
                     "top10_overlap": "", "EN_R10": recall_ks(ixg, pack_bits(bf16_codes), gold)[10],
                     "KO_R10": "", "note": "so400m tower bf16 vs fp32 backbone, head fp32 (EN)"})
    return rows


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_img = EC["test"]["img"].float().numpy()
    en_orig = EC["test"]["txt"].float().numpy()                       # exact 79.92 cache
    en_low = torch.load("/tmp/coco_en_lc.pt", map_location="cpu").float().numpy()
    ko = KO["txt_emb"].float().numpy()                                # caseless
    en_bf16 = en_low                                                   # bf16-backbone lowercased
    en_low_fp32 = torch.load("/tmp/coco_en_lc_fp32.pt", map_location="cpu").float().numpy() if os.path.exists("/tmp/coco_en_lc_fp32.pt") else None
    en_orig_fp32 = torch.load("/tmp/coco_en_orig_fp32.pt", map_location="cpu").float().numpy() if os.path.exists("/tmp/coco_en_orig_fp32.pt") else None
    gold = list(range(te_img.shape[0]))

    all_rows = []
    for variant, en, enfp in (("orig", en_orig, en_orig_fp32), ("lower", en_low, en_low_fp32)):
        rows = run_variant(enc, te_img, {"EN": en, "KO": ko}, gold,
                           en_bf16=(en_low if variant == "lower" else en_orig), en_fp32=enfp)
        for r in rows:
            r["variant"] = variant
            all_rows.append(r)
        print(f"[prec] variant={variant}:", flush=True)
        for r in rows:
            print(f"[prec]   {r['target']}/{r['dtype']}: flip {r['bitflips_per_1024']}/1024 "
                  f"overlap {r['top10_overlap']} EN_R10 {r['EN_R10']} KO_R10 {r['KO_R10']} {r['note']}", flush=True)

    with open(PAPER / "precision_lc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "dtype", "variant", "bitflips_per_1024", "top10_overlap",
                                          "EN_R10", "KO_R10", "note"], extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)

    def g(target, dtype, variant, key):
        for r in all_rows:
            if r["target"] == target and r["dtype"] == dtype and r["variant"] == variant:
                return r.get(key)
        return None
    print("[prec] VALIDATE orig vs recorded precision.csv: "
          f"fp32 EN {g('head','fp32','orig','EN_R10')}(79.92) | head fp16 {g('head','fp16','orig','bitflips_per_1024')}(0.12) "
          f"bf16 {g('head','bf16','orig','bitflips_per_1024')}(0.95) int8 {g('head','int8','orig','bitflips_per_1024')}(18.97) | "
          f"emb int8 {g('emb','int8','orig','bitflips_per_1024')}(5.45) | text_tower {g('text_tower','bf16','orig','bitflips_per_1024')}(2.57)", flush=True)
    for tgt, dt in (("head","fp16"),("head","bf16"),("head","int8"),("emb","fp16"),("emb","bf16"),("emb","int8"),("text_tower","bf16")):
        print(f"[prec] CASE-INV {tgt}/{dt}: flips orig {g(tgt,dt,'orig','bitflips_per_1024')} ~ lower "
              f"{g(tgt,dt,'lower','bitflips_per_1024')} | EN_R10 {g(tgt,dt,'orig','EN_R10')} -> {g(tgt,dt,'lower','EN_R10')}", flush=True)
    print("[prec] RESULT_JSON " + json.dumps({"rows": all_rows}, ensure_ascii=False), flush=True)
    print(f"[prec] DONE -> paper/precision_lc.csv ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
