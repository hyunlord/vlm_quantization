"""Aggregate paper/nvs_runs/*.json (from exp_nested_vs_single.py) into CSVs.

Outputs:
  paper/nested_vs_single64.csv          arm, seed, metric, value          (long format, every run)
  paper/nested_vs_single64_summary.csv  arm, metric, mean, std, n         (std = sample std, ddof=1)
  paper/nested_prefix_consistency.csv   arm, seed, split, sample_exact_match, bit_agreement, n
  paper/nested_vs_single64_compare.csv  a, b, metric, mean_a, std_a, mean_b, std_b, diff(a-b), threshold(std_a+std_b), verdict

Verdict rule (brief): |mean_a - mean_b| > std_a + std_b -> "difference", else "no difference (seed noise)".
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "paper" / "nvs_runs"
PAIRS = [("S64", "N1024"), ("N64", "N1024"), ("S64", "N64"), ("N1024", "N1024-noLCS")]


def main():
    runs = [json.load(open(p)) for p in sorted(RUNS.glob("*_s*.json"))]
    long_rows, by = [], {}
    for r in runs:
        vals = dict(r["metrics"]); vals["train_sec"] = r["train_sec"]
        for k, v in vals.items():
            long_rows.append((r["arm"], r["seed"], k, v)); by.setdefault((r["arm"], k), []).append(v)
    with open(REPO / "paper" / "nested_vs_single64.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "seed", "metric", "value"]); w.writerows(long_rows)

    summ = {}
    with open(REPO / "paper" / "nested_vs_single64_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "metric", "mean", "std", "n"])
        for (arm, k), v in sorted(by.items()):
            m = st.mean(v); s = st.stdev(v) if len(v) > 1 else float("nan")
            summ[(arm, k)] = (m, s, len(v)); w.writerow([arm, k, m, s, len(v)])
        fc = RUNS / "float_cosine.json"
        if fc.exists():
            for k, v in json.load(open(fc))["metrics"].items():
                w.writerow(["float_cosine", k, v, "", 1])

    with open(REPO / "paper" / "nested_prefix_consistency.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "seed", "split", "sample_exact_match", "bit_agreement", "n"])
        for r in runs:
            for split, d in (r.get("prefix_consistency") or {}).items():
                w.writerow([r["arm"], r["seed"], split, d["sample_exact_match"], d["bit_agreement"], d["n"]])

    arms = {r["arm"] for r in runs}; metrics = sorted({k for (_, k) in by})
    with open(REPO / "paper" / "nested_vs_single64_compare.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a", "b", "metric", "mean_a", "std_a", "mean_b", "std_b", "diff", "threshold", "verdict"])
        for a, b in PAIRS:
            if a not in arms or b not in arms:
                continue
            for k in metrics:
                if (a, k) not in summ or (b, k) not in summ:
                    continue
                (ma, sa, _), (mb, sb, _) = summ[(a, k)], summ[(b, k)]
                d = ma - mb; th = sa + sb
                w.writerow([a, b, k, ma, sa, mb, sb, d, th, "difference" if abs(d) > th else "no difference (seed noise)"])
    print(f"runs={len(runs)} arms={sorted(arms)} -> paper/nested_vs_single64*.csv, nested_prefix_consistency.csv")


if __name__ == "__main__":
    main()
