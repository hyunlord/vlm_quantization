"""(ii) Multiling table re-derived with the so400m lowercase fix + rescope the offline-wins claim.

so400m text tower: LOWERCASED (from paper/german_sanity.csv — all 36 langs, the verified fix; de 96.54,
avg36 74.46). Offline encoders (MiniLM/e5) and NLLB/AltCLIP: NATIVE preprocessing (unchanged — model-specific,
case-preserving; their numbers come from /tmp/multiling.json (Ext②) and paper/baselines_multiling.csv).

Key recompute: with so400m now correctly lowercased, on which langs does the OFFLINE 1-bit head still beat
the so400m text tower? Expectation: only non-Latin low-resource scripts (te/th/hi/bn), since lowercasing
fixed all cased Latin/Cyrillic langs. No GPU — pure recomputation from cached results.

Run on DGX: .venv/bin/python web/paper_multiling_lc.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
PAPER = Path(REPO) / "paper"
MJSON = os.environ.get("MULTILING_JSON", "/tmp/multiling.json")
NON_LATIN = {"te", "th", "hi", "bn", "ja", "ko", "zh", "ar", "fa", "he", "el", "ru", "uk", "ta", "ml", "kn"}


def main():
    # so400m text tower lowercased + orig from german_sanity.csv
    so_lower, so_orig = {}, {}
    for row in csv.DictReader(open(PAPER / "german_sanity.csv")):
        L = row["lang"]
        if L in ("AVG36",):
            continue
        if row["variant"].startswith("baseline"):
            so_orig[L] = (float(row["R10"]), row.get("mAP10") or "")
        elif row["variant"] == "lowercased":
            so_lower[L] = (float(row["R10"]), row.get("mAP10") or "")

    # offline encoders (native) from Ext② multiling.json
    mj = json.load(open(MJSON)) if os.path.exists(MJSON) else {"data": {}}
    data = mj.get("data", {})
    offline_keys = [k for k in data if ("MiniLM" in k or "e5" in k)]
    server_key = next((k for k in data if "server" in k), None)

    rows, wins = [], []
    langs = sorted(so_lower.keys())
    for L in langs:
        rows.append({"model": "so400m text tower", "preproc": "lowercased(FIX)", "lang": L,
                     "R10": so_lower[L][0], "mAP10": so_lower[L][1]})
        rows.append({"model": "so400m text tower", "preproc": "orig-case(buggy)", "lang": L,
                     "R10": so_orig.get(L, ("", ""))[0], "mAP10": ""})
        for k in offline_keys:
            if L in data[k]:
                rows.append({"model": f"offline {k}", "preproc": "native(1bit head)", "lang": L,
                             "R10": data[k][L]["10"], "mAP10": ""})
        if server_key and L in data[server_key]:
            rows.append({"model": "server so400m+ft113", "preproc": "orig-case(buggy,1bit)", "lang": L,
                         "R10": data[server_key][L]["10"], "mAP10": ""})
        # offline-wins: best offline (native) vs so400m text tower LOWERCASED
        best_off = max((data[k][L]["10"] for k in offline_keys if L in data[k]), default=None)
        if best_off is not None and best_off > so_lower[L][0]:
            wins.append((L, round(best_off - so_lower[L][0], 2), "non-Latin" if L in NON_LATIN else "LATIN(!)"))

    with open(PAPER / "multiling_lc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "preproc", "lang", "R10", "mAP10"])
        w.writeheader(); w.writerows(rows)
    win_langs = sorted(w[0] for w in wins)
    print(f"[ii] offline-wins langs (best offline native > so400m-lowercased): {win_langs}", flush=True)
    print(f"[ii] all non-Latin? {all(L in NON_LATIN for L in win_langs)} | details {wins}", flush=True)
    print("[ii] RESULT_JSON " + json.dumps({"offline_wins": win_langs, "wins_detail": wins,
          "so400m_avg36_lower": round(sum(v[0] for v in so_lower.values())/len(so_lower), 2)}, ensure_ascii=False), flush=True)
    print(f"[ii] DONE -> paper/multiling_lc.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
