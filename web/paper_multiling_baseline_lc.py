"""(a2) Multiling baseline table (batch #1 (a) baselines_multiling.csv) — so400m rows lowercased.

Pure re-aggregation (NO GPU, NO model load): the lowercased so400m values already exist in the corrected
CSVs from batches #2–#4:
  so400m FLOAT text tower lowercased  -> german_sanity.csv (XM3600, all langs) + coco_lc_full.csv (COCO float_raw)
  so400m+ft113 SERVER 1-bit lowercased -> multiling_server_lc.csv (XM3600 server_lc_R10) + coco_lc_full.csv (server)
value_orig comes from baselines_multiling.csv (the orig-case batch #1 (a) table).
NLLB-CLIP / AltCLIP / MiniLM rows keep NATIVE preprocessing (unchanged) — emitted as a reminder, value_lower
== value_orig. Appends to paper/encoders_baselines_lc.csv (written first by paper_siglip2base_alt_lc.py).

Run: .venv/bin/python web/paper_multiling_baseline_lc.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
PAPER = Path(REPO) / "paper"
OUT = PAPER / "encoders_baselines_lc.csv"
COLS = ["table", "backbone", "lang_or_metric", "value_orig", "value_lower", "preproc"]
REPR_XM = ["de", "en", "hi", "ko", "te", "th", "avg36"]


def read_csv(name):
    p = PAPER / name
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def main():
    bm = read_csv("baselines_multiling.csv")
    gs = read_csv("german_sanity.csv")          # so400m float tower lowercased (R1/R5/R10/mAP10)
    ms = read_csv("multiling_server_lc.csv")     # ceiling_lc + server_lc_R10 (lowercased)
    cf = read_csv("coco_lc_full.csv")            # COCO so400m server/float_raw orig+lower

    # --- index orig values from baselines_multiling.csv ---
    bm_o = {}  # (model, dataset, lang) -> R10
    for r in bm:
        bm_o[(r["model"], r["dataset"], r["lang"])] = r["R10"]
    # --- lowercased lookups ---
    gs_lower = {r["lang"]: r["R10"] for r in gs if r["variant"] == "lowercased"}  # XM float-lower R10
    gs_avg = next((r["R10"] for r in gs if r["lang"] == "AVG36" and r["variant"] == "lowercased"), "")
    gs_lower["avg36"] = gs_avg
    ms_ceiling = {r["lang"]: r["ceiling_lc"] for r in ms}     # == so400m float lower R10 (XM)
    ms_server = {r["lang"]: r["server_lc_R10"] for r in ms}   # so400m+ft113 server lower R10 (XM)
    ms_ceiling["avg36"] = ms_ceiling.get("AVG36", "")         # multiling_server_lc uses "AVG36"
    ms_server["avg36"] = ms_server.get("AVG36", "")
    # COCO so400m lower from coco_lc_full
    cf_lower = {}  # (path, lang) -> R10 ; path in {server, float_raw}
    for r in cf:
        if r["preproc"] == "lower" and r["row"].startswith("so400m|"):
            cf_lower[(r["row"].split("|")[1], r["lang"])] = r["R10"]

    rows = []

    def emit(backbone, metric, vorig, vlower, preproc):
        rows.append({"table": "baselines_multiling", "backbone": backbone, "lang_or_metric": metric,
                     "value_orig": vorig, "value_lower": vlower, "preproc": preproc})

    FLOAT = "SigLIP2 text tower (so400m float)"
    SERVER = "Ours: server (so400m+ft113)"
    # COCO en/ko
    for L in ("en", "ko"):
        emit("so400m-float", f"coco_{L}_R10", bm_o.get((FLOAT, "coco", L), ""),
             cf_lower.get(("float_raw", L.upper()), ""), "lowercased")
        emit("so400m-server-1bit", f"coco_{L}_R10", bm_o.get((SERVER, "coco", L), ""),
             cf_lower.get(("server", L.upper()), ""), "lowercased")
    # XM3600 REPR + avg36
    for L in REPR_XM:
        emit("so400m-float", f"xm3600_{L}_R10", bm_o.get((FLOAT, "xm3600", L), ""),
             gs_lower.get(L, ms_ceiling.get(L, "")), "lowercased")
        emit("so400m-server-1bit", f"xm3600_{L}_R10", bm_o.get((SERVER, "xm3600", L), ""),
             ms_server.get(L, ""), "lowercased")
    # native rows (unchanged) — emit as explicit reminders
    for model in ("NLLB-CLIP (nllb-clip-base-siglip)", "AltCLIP-m18", "Ours: offline (MiniLM head-adapt)"):
        for L in ("en", "avg36"):
            v = bm_o.get((model, "xm3600", L)) or bm_o.get((model, "coco", L))
            if v is not None:
                tag = "native(unchanged: not SigLIP-family)" if "MiniLM" not in model else "native(offline, case-insensitive)"
                emit(model, f"xm3600_{L}_R10" if bm_o.get((model, "xm3600", L)) else f"coco_{L}_R10", v, v, tag)

    # idempotent: keep existing non-baseline (backbone) rows, replace the baselines_multiling block
    keep = [r for r in (csv.DictReader(open(OUT, encoding="utf-8")) if OUT.exists() else [])
            if r.get("table") != "baselines_multiling"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader(); w.writerows(keep); w.writerows(rows)

    # snapshot print
    print("[a2] so400m multiling baseline rows lowercased (XM3600 R@10, orig->lower):", flush=True)
    for r in rows:
        if r["backbone"].startswith("so400m") and "xm3600" in r["lang_or_metric"]:
            print(f"[a2]   {r['backbone']:>20} {r['lang_or_metric']:<16} {r['value_orig']} -> {r['value_lower']}", flush=True)
    print(f"[a2] ANCHORS: float de 37.58->{gs_lower.get('de')} avg36 66.89->{gs_lower.get('avg36')} | "
          f"server de 30.05->{ms_server.get('de')} avg36 62.36->{ms_server.get('avg36')}", flush=True)
    print(f"[a2] DONE -> appended {len(rows)} rows to paper/encoders_baselines_lc.csv", flush=True)


if __name__ == "__main__":
    main()
