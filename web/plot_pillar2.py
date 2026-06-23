"""Render Pillar-2 margin figures (PDF) from the CSVs. Run: .venv/bin/python web/plot_pillar2.py"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rd(p):
    return list(csv.DictReader(open(p)))


# (2) flip vs |z|: empirical vs Φ(−|z|/σ)
d = rd("paper/pillar2_flip_vs_margin.csv")
x = [float(r["absz_bin"]) for r in d]
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.scatter(x, [float(r["emp_flip"]) for r in d], s=[float(r["n"])/3000 for r in d], c="tab:blue", label="empirical", zorder=3)
ax.plot(x, [float(r["gauss_pred"]) for r in d], "r-", label=r"$\Phi(-|z|/\sigma)$", zorder=2)
ax.set_xlabel("pre-sign margin |z|"); ax.set_ylabel("sign-flip prob (int8 noise)")
ax.set_title("flip vs margin (weighted $R^2$=0.81)"); ax.legend(); fig.tight_layout()
fig.savefig("paper/fig_pillar2_flip_vs_margin.pdf"); plt.close(fig)

# (3) gate: D vs R10 and parity vs R10 (none predicts)
g = rd("paper/pillar2_margin_predicts_R.csv")
col = {"ceiling": "tab:green", "distillation": "tab:red", "head-adapt": "tab:blue"}
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, key, lab in ((axs[0], "D_expected_distort", "D = mean Φ(−|z|/σ)"),
                     (axs[1], "parity_pct", "bit-parity vs true (%)"),
                     (axs[2], "mean_absz", "mean |z|")):
    for r in g:
        ax.scatter(float(r[key]), float(r["binR10"]), c=col[r["path"]], s=40 + float(r["bit"]) / 20)
    ax.set_xlabel(lab); ax.set_ylabel("binary R@10")
axs[0].set_title("margin D vs R@10 (Spearman 0.25, wrong sign)")
axs[1].set_title("parity vs R@10 (0.32)")
axs[2].set_title("mean|z| vs R@10 (−0.30)")
hs = [plt.Line2D([], [], marker="o", ls="", color=c, label=p) for p, c in col.items()]
axs[2].legend(handles=hs, fontsize=8)
fig.suptitle("Gate: no margin/parity stat predicts retrieval (size ∝ bits)")
fig.tight_layout(); fig.savefig("paper/fig_pillar2_margin_predicts_R.pdf"); plt.close(fig)

# (3) per-query success by D_q quintile (flat)
q = rd("paper/pillar2_perquery.csv")
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.bar([int(r["Dq_quintile"]) for r in q], [float(r["success_rate"]) for r in q], color="tab:gray")
ax.set_xlabel("per-query expected distortion $D_q$ quintile (1=low)"); ax.set_ylabel("R@10 success rate (%)")
ax.set_ylim(70, 90); ax.set_title("per-query margin does not predict success (flat)")
fig.tight_layout(); fig.savefig("paper/fig_pillar2_perquery.pdf"); plt.close(fig)
print("wrote paper/fig_pillar2_*.pdf")
