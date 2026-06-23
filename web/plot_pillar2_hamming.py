"""Pillar2 Hamming-predictor figures (PDF). Run: .venv/bin/python web/plot_pillar2_hamming.py"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = list(csv.DictReader(open("paper/pillar2_hamming_predictors.csv")))
col = {"ceiling": "tab:green", "distillation": "tab:red", "head-adapt": "tab:blue"}
preds = [("jaccard@100", "neighbor Jaccard (ρ=0.70, paradox WRONG)"),
         ("diralign_mean", "direction-align (ρ=−0.48, paradox WRONG)"),
         ("parity_pct", "parity (ρ=0.14, WRONG)"),
         ("hmargin_frac_pos", "Hamming-margin frac>0 (ρ=1.00, TAUTOLOGICAL)")]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (k, title) in zip(axs, preds):
    for r in g:
        ax.scatter(float(r[k]), float(r["binR10"]), c=col[r["path"]], s=40 + float(r["bit"]) / 20, zorder=3)
    ax.set_xlabel(k); ax.set_ylabel("binary R@10"); ax.set_title(title, fontsize=9)
hs = [plt.Line2D([], [], marker="o", ls="", color=c, label=p) for p, c in col.items()]
axs[0].legend(handles=hs, fontsize=8)
fig.suptitle("Pillar2 gate v2: structural proxies fail/invert; only the tautological (retrieval-restating) one is monotone (size ∝ bits)")
fig.tight_layout(); fig.savefig("paper/fig_pillar2_hamming_predictors.pdf"); plt.close(fig)
print("wrote paper/fig_pillar2_hamming_predictors.pdf")
