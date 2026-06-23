"""Plot Part 1: each candidate predictor vs R@10 (cross-language), Spearman annotated."""
from __future__ import annotations
import csv, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.environ.get("REPO", "."), "paper")
rows = [r for r in csv.DictReader(open(os.path.join(OUT, "rank_predictor.csv"))) if r["axis"] == "lang"]
y = np.array([float(r["r10"]) for r in rows])
VARS = [("upstream_effdim", "upstream text-emb eff-dim (actionable)"),
        ("code_effrank", "code effective-rank (downstream)"),
        ("code_entropy", "code bit-entropy (downstream/tautology-risk)"),
        ("cosine", "pair cosine (proxy, embedding)"),
        ("margin", "mean|z| (proxy; ~constant)"),
        ("diralign", "code dir-align (proxy; pair-Hamming = tautology)")]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (v, lab) in zip(axes.ravel(), VARS):
    x = np.array([float(r[v]) for r in rows]); rho = spearmanr(x, y).correlation
    for r, xi, yi in zip(rows, x, y):
        c = "#d95f02" if r["script"] == "non-latin" else "#1b9e77"
        ax.scatter(xi, yi, color=c, s=26)
        if r["key"] in ("en", "ko", "mi", "te", "quz", "sv", "fil", "de"):
            ax.annotate(r["key"], (xi, yi), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"{lab}\nSpearman ρ = {rho:+.3f}", fontsize=9); ax.set_xlabel(v); ax.set_ylabel("R@10")
axes[0][0].scatter([], [], color="#1b9e77", label="latin"); axes[0][0].scatter([], [], color="#d95f02", label="non-latin"); axes[0][0].legend(fontsize=8)
fig.suptitle("Part 1: candidate predictors of binary R@10 across 36 languages (XM3600, ceiling path)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rank_predictors.pdf")); plt.close(fig)
print("  wrote paper/fig_rank_predictors.pdf", flush=True)
