"""(C) Master summary table — every backbone/config on one footing, correct (lowercased so400m) preprocessing.

Re-aggregates the corrected CSVs into one paper/master_table.csv. NO new compute. Recipe/preproc tags are
mandatory: ft113-full (so400m, aug+OI+RKD+KO-finetune) vs clean-EN(+ko) head-only are NOT directly value-
comparable — the table is for POSITIONING/patterns, stated in the caption. Blank cells are left empty and
listed under "needs-measure".

Sources: coco_lc_full.csv, multiling_server_lc.csv (so400m server lower), german_sanity.csv (so400m float
lower avg36), baseline_lc.csv (offline e5/MiniLM native), backbones.csv + encoders_baselines_lc.csv (base/
AltCLIP), nllb_hashing.csv, metaclip2_hashing.csv (B), multiling.json (offline xm36 avg36).

Run (after a1, a2, B): .venv/bin/python web/paper_master_table.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
PAPER = Path(REPO) / "paper"
MJSON = os.environ.get("MULTILING_JSON", "/tmp/multiling.json")
COLS = ["backbone", "family", "params_M", "text_path", "recipe", "bits", "bytes_per_img",
        "deployable_browser", "coco_en_R10", "coco_ko_R10", "coco_en_mAP10", "xm3600_avg36_R10", "source_csv"]


def rd(name):
    p = PAPER / name
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def main():
    needs = []
    cf = rd("coco_lc_full.csv")
    ms = rd("multiling_server_lc.csv")
    bl = rd("baseline_lc.csv")
    bb = {r["backbone"]: r for r in rd("backbones.csv")}
    eb = rd("encoders_baselines_lc.csv")
    nllb = rd("nllb_hashing.csv")
    mc = rd("metaclip2_hashing.csv")
    mj = json.load(open(MJSON)) if os.path.exists(MJSON) else {"data": {}}

    def cf_get(row_path, lang, preproc, key):
        for r in cf:
            if r["row"] == f"so400m|{row_path}" and r["lang"] == lang and r["preproc"] == preproc:
                return r[key]
        return ""

    def ms_avg36():
        for r in ms:
            if r["lang"] == "AVG36":
                return r["server_lc_R10"]
        return ""

    def gs_avg36():
        for r in rd("german_sanity.csv"):
            if r["lang"] == "AVG36" and r["variant"] == "lowercased":
                return r["R10"]
        return ""

    def eb_get(backbone, metric):
        for r in eb:
            if r["backbone"] == backbone and r["lang_or_metric"] == metric:
                return r["value_lower"]
        return ""

    def bl_get(model, lang, key):
        for r in bl:
            if r["model"] == model and r["lang"] == lang and r.get("variant") in ("native", None):
                return r[key]
        return ""

    def offline_xm_avg(key_hint):
        data = mj.get("data", {})
        k = next((k for k in data if key_hint in k), None)
        if not k:
            return ""
        vals = [data[k][L]["10"] for L in data[k] if isinstance(data[k][L], dict) and "10" in data[k][L]]
        return round(float(np.mean(vals)), 2) if vals else ""

    def hashing_get(rows, model, dataset, lang, key):
        for r in rows:
            if r["model"] == model and r["dataset"] == dataset and r["lang"] == lang:
                return r[key]
        return ""

    rows = []

    def add(backbone, family, params, text_path, recipe, bits, bpi, deploy, ce, ck, cm, xm, src):
        for label, v in (("coco_en_R10", ce), ("coco_ko_R10", ck), ("xm3600_avg36_R10", xm)):
            if v == "" or v is None:
                needs.append(f"{backbone}/{text_path}/{recipe}: {label}")
        rows.append({"backbone": backbone, "family": family, "params_M": params, "text_path": text_path,
                     "recipe": recipe, "bits": bits, "bytes_per_img": bpi, "deployable_browser": deploy,
                     "coco_en_R10": ce, "coco_ko_R10": ck, "coco_en_mAP10": cm, "xm3600_avg36_R10": xm,
                     "source_csv": src})

    # ---- SigLIP2-So400m + ft113 ----
    add("SigLIP2-So400m+ft113", "SigLIP2-so400m", 707.8, "server", "ft113-full", 1024, 128, "N",
        cf_get("server", "EN", "lower", "R10"), cf_get("server", "KO", "lower", "R10"),
        cf_get("server", "EN", "lower", "mAP10"), ms_avg36(), "coco_lc_full,multiling_server_lc")
    add("SigLIP2-So400m float (ceiling)", "SigLIP2-so400m", 707.8, "server", "float-ceiling", "-", "-", "N",
        cf_get("float_raw", "EN", "lower", "R10"), cf_get("float_raw", "KO", "lower", "R10"),
        cf_get("float_raw", "EN", "lower", "mAP10"), gs_avg36(), "coco_lc_full,german_sanity")
    add("SigLIP2-So400m naive-sign", "SigLIP2-so400m", 707.8, "server", "naive(no head)", 1152, 144, "N",
        cf_get("naive", "EN", "lower", "R10"), cf_get("naive", "KO", "lower", "R10"),
        cf_get("naive", "EN", "lower", "mAP10"), "", "coco_lc_full")
    # offline (browser-deployable small encoders) — native preprocessing
    add("SigLIP2-So400m+ft113 -> e5-small", "e5/XLM-R", 117.7, "offline", "head-adapt(clean)", 1024, 128, "Y",
        bl_get("e5-small", "EN", "R10"), bl_get("e5-small", "KO", "R10"), bl_get("e5-small", "EN", "mAP10"),
        offline_xm_avg("e5-small"), "baseline_lc,multiling.json")
    add("SigLIP2-So400m+ft113 -> MiniLM", "paraphrase/XLM-R", 117.7, "offline", "head-adapt(clean)", 1024, 128, "Y",
        bl_get("MiniLM", "EN", "R10"), bl_get("MiniLM", "KO", "R10"), bl_get("MiniLM", "EN", "mAP10"),
        offline_xm_avg("MiniLM"), "baseline_lc,multiling.json")

    # ---- SigLIP2-base + head (clean-EN), lowercased (a1) ----
    add("SigLIP2-base+head", "SigLIP2-base", "", "server", "clean-EN", 1024, 128, "N",
        eb_get("SigLIP2-base", "server_1bit_EN_R10"), eb_get("SigLIP2-base", "server_1bit_KO_R10"),
        "", "", "encoders_baselines_lc(backbones)")
    # ---- AltCLIP + head (clean-EN), native ----
    add("AltCLIP-m18+head", "AltCLIP-XLMR", "", "server", "clean-EN(native)", 1024, 128, "N",
        bb.get("BAAI/AltCLIP-m18", {}).get("server_1bit_EN_R10", ""),
        bb.get("BAAI/AltCLIP-m18", {}).get("server_1bit_KO_R10", ""), "", "", "backbones")
    # ---- NLLB-CLIP + head (clean, native) ----
    add("NLLB-CLIP+head", "NLLB-CLIP", "", "server", "clean-EN+ko(native)", 1024, 128, "N",
        hashing_get(nllb, "NLLB+head", "coco", "en", "R10"), hashing_get(nllb, "NLLB+head", "coco", "ko", "R10"),
        hashing_get(nllb, "NLLB+head", "coco", "en", "mAP10"),
        hashing_get(nllb, "NLLB+head", "xm3600", "avg36", "R10"), "nllb_hashing")
    add("NLLB-CLIP float (ceiling)", "NLLB-CLIP", "", "server", "float-ceiling", "-", "-", "N",
        hashing_get(nllb, "NLLB float", "coco", "en", "R10"), hashing_get(nllb, "NLLB float", "coco", "ko", "R10"),
        hashing_get(nllb, "NLLB float", "coco", "en", "mAP10"),
        hashing_get(nllb, "NLLB float", "xm3600", "avg36", "R10"), "nllb_hashing")
    # ---- MetaCLIP2 + head (clean, native) — batch #5 (B) ----
    add("MetaCLIP2+head", "MetaCLIP2-worldwide", "", "server", "clean-EN+ko(native)", 1024, 128, "N",
        hashing_get(mc, "MetaCLIP2+head", "coco", "en", "R10"), hashing_get(mc, "MetaCLIP2+head", "coco", "ko", "R10"),
        hashing_get(mc, "MetaCLIP2+head", "coco", "en", "mAP10"),
        hashing_get(mc, "MetaCLIP2+head", "xm3600", "avg36", "R10"), "metaclip2_hashing")
    add("MetaCLIP2 float (ceiling)", "MetaCLIP2-worldwide", "", "server", "float-ceiling", "-", "-", "N",
        hashing_get(mc, "MetaCLIP2 float", "coco", "en", "R10"), hashing_get(mc, "MetaCLIP2 float", "coco", "ko", "R10"),
        hashing_get(mc, "MetaCLIP2 float", "coco", "en", "mAP10"),
        hashing_get(mc, "MetaCLIP2 float", "xm3600", "avg36", "R10"), "metaclip2_hashing")

    with open(PAPER / "master_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"[C] master_table.csv: {len(rows)} rows", flush=True)
    for r in rows:
        print(f"[C]   {r['backbone']:<34} {r['text_path']:<7} {r['recipe']:<22} "
              f"coco EN/KO {r['coco_en_R10']}/{r['coco_ko_R10']} xm36 {r['xm3600_avg36_R10']} [{r['source_csv']}]", flush=True)
    print(f"[C] needs-measure ({len(needs)}): {needs}", flush=True)
    print(f"[C] DONE -> paper/master_table.csv", flush=True)


if __name__ == "__main__":
    main()
