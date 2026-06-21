"""(2) Multiling SERVER (so400m+ft113, 1-bit) column lowercased + confirm the "offline beats backbone" set.

Batch #3 lowercased only the multiling FLOAT ceiling (german_sanity.csv). The server (so400m+ft113, 1-bit)
column in multiling.json is still ORIG-CASE (de 30.05 = the casing bug). This re-derives the 36-lang server
1-bit R@10 (+ mAP@10) with the so400m text LOWERCASED (ft113 head applied to /tmp/xm_so400m_lc.pt, built by
paper_coco_lc_full.py), then joins:
  ceiling_lc   = so400m text-tower float R@10 lowercased  (german_sanity.csv, anchor de 96.54 / avg36 74.46)
  offline_best = best deployable offline 1-bit head, NATIVE preprocessing (multiling.json: e5-small/MiniLM/e5-base)
and recomputes beats_backbone = offline_best > ceiling_lc  (offline beats even the so400m FLOAT ceiling).

Expectation (work order): once so400m is correctly lowercased, offline only wins on non-Latin low-resource
scripts (te/th/hi, ~bn); cased langs (de) flip back to backbone-wins. No backbone reload (head-only on cached
lowercased embeddings).

Run: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_multiling_server_lc.py
"""
from __future__ import annotations

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

PAPER = Path(REPO) / "paper"
MJSON = os.environ.get("MULTILING_JSON", "/tmp/multiling.json")
NON_LATIN = {"te", "th", "hi", "bn", "ja", "ko", "zh", "ar", "fa", "he", "el", "ru", "uk", "ta", "ml", "kn"}
OFFLINE_KEYS_HINT = ("e5-small", "MiniLM", "e5-base")


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def rk(ix, q_packed, gold):
    _, I = ix.search(np.ascontiguousarray(q_packed), 10)
    r10 = round(100 * float(np.mean([gold[i] in I[i, :10] for i in range(len(gold))])), 2)
    ap = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0)
          for i in range(len(gold))]
    return r10, round(100 * float(np.mean(ap)), 2)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    xm = torch.load("/tmp/xm_so400m_lc.pt", map_location="cpu")  # img_emb + per_lang lowercased
    enc = Encoder()
    gal = enc.image_codes_packed(xm["img_emb"].float().numpy()); ix = faiss_bin(gal, enc.bits)

    # ceiling_lc (so400m float tower, lowercased) from german_sanity.csv
    ceiling = {}
    gpath = PAPER / "german_sanity.csv"
    for row in csv.DictReader(open(gpath)):
        if row["variant"] == "lowercased" and row["lang"] not in ("AVG36",):
            ceiling[row["lang"]] = float(row["R10"])

    # offline (native) from multiling.json
    mj = json.load(open(MJSON)) if os.path.exists(MJSON) else {"data": {}}
    data = mj.get("data", {})
    offline_keys = [k for k in data if any(h in k for h in OFFLINE_KEYS_HINT)]

    rows, wins, beats_server = [], [], []
    langs = sorted(xm["per_lang"].keys())
    s_r10s = []
    for L in langs:
        d = xm["per_lang"][L]
        sr, sm = rk(ix, pack_bits(enc._codes_pm1(enc.txt_h, d["text_emb"])), d["gold"])
        s_r10s.append(sr)
        cl = ceiling.get(L, "")
        offs = [(k, data[k][L]["10"]) for k in offline_keys if L in data[k]]
        best_k, best_off = ("", "")
        if offs:
            best_k, best_off = max(offs, key=lambda kv: kv[1])
        bb_ceiling = isinstance(best_off, (int, float)) and isinstance(cl, (int, float)) and best_off > cl
        bb_server = isinstance(best_off, (int, float)) and best_off > sr
        if bb_ceiling:
            wins.append((L, round(best_off - cl, 2), "non-Latin" if L in NON_LATIN else "LATIN(!)"))
        if bb_server:
            beats_server.append(L)
        rows.append({"lang": L, "ceiling_lc": cl, "server_lc_R10": sr, "server_lc_mAP10": sm,
                     "offline_best": best_off, "offline_model": best_k,
                     "beats_backbone": bool(bb_ceiling), "beats_server": bool(bb_server)})
        print(f"[ii] {L}: ceiling_lc {cl} | server_lc {sr} (mAP {sm}) | offline_best {best_off} ({best_k}) "
              f"| beats_ceiling {bb_ceiling} beats_server {bb_server}", flush=True)

    server_avg = round(float(np.mean(s_r10s)), 2)
    ceil_avg = round(float(np.mean([ceiling[L] for L in langs if L in ceiling])), 2)
    rows.append({"lang": "AVG36", "ceiling_lc": ceil_avg, "server_lc_R10": server_avg, "server_lc_mAP10": "",
                 "offline_best": "", "offline_model": "", "beats_backbone": "", "beats_server": ""})

    with open(PAPER / "multiling_server_lc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "ceiling_lc", "server_lc_R10", "server_lc_mAP10",
                                          "offline_best", "offline_model", "beats_backbone", "beats_server"])
        w.writeheader(); w.writerows(rows)
    win_langs = sorted(w0[0] for w0 in wins)
    print(f"[ii] AVG36 ceiling_lc {ceil_avg} (anchor 74.46) | server_lc {server_avg}", flush=True)
    print(f"[ii] offline beats so400m FLOAT ceiling on: {win_langs} | all non-Latin? "
          f"{all(L in NON_LATIN for L in win_langs)} | detail {wins}", flush=True)
    print(f"[ii] offline beats so400m+ft113 SERVER(1bit) on: {sorted(beats_server)}", flush=True)
    print("[ii] RESULT_JSON " + json.dumps({"server_avg36": server_avg, "ceiling_avg36": ceil_avg,
          "offline_wins_vs_ceiling": win_langs, "wins_detail": wins, "beats_server": sorted(beats_server)},
          ensure_ascii=False), flush=True)
    print(f"[ii] DONE -> paper/multiling_server_lc.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
