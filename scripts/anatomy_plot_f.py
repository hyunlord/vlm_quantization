"""Plot anatomy F: F1 cumulative variance + residual; F2 code-entropy & effdim vs R@10."""
from __future__ import annotations
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.environ.get("REPO", "."), "paper")
COL = {"image": "#444444", "ceiling": "#1b9e77", "distillation": "#d95f02"}


def load(n):
    with open(os.path.join(OUT, n)) as f:
        return list(csv.DictReader(f))


def save(fig, n):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, n)); plt.close(fig); print(f"  wrote paper/{n}", flush=True)


# F1 cumulative variance
cv = defaultdict(list)
for r in load("anatomy_F1_cumvar.csv"):
    cv[r["source"]].append((int(r["k"]), float(r["cum_var_frac"])))
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for s in ["image", "ceiling", "distillation"]:
    k, c = zip(*sorted(cv[s])); ax.plot(k, c, "o-", color=COL[s], label=s)
ax.axhline(0.9, color="k", lw=0.5, ls=":"); ax.set_xscale("log")
ax.set_xlabel("# top PCA components (k)"); ax.set_ylabel("cumulative variance fraction")
ax.set_title("F1: cumulative explained variance (backbone embeddings)"); ax.legend(fontsize=8)
save(fig, "fig_anatomy_F1_cumvar.pdf")

# F1 residual energy
rr = load("anatomy_F1_residual.csv")
fig, ax = plt.subplots(figsize=(6.2, 4.3))
labels = sorted(set((r["projected"], r["onto_basis_of"]) for r in rr))
for (a, b) in labels:
    pts = [(int(r["k"]), float(r["energy_in_subspace_frac"])) for r in rr if r["projected"] == a and r["onto_basis_of"] == b]
    k, e = zip(*sorted(pts)); ax.plot(k, e, "o-", label=f"{a} in {b} PCs")
ax.set_xlabel("# top PCs of basis (k)"); ax.set_ylabel("energy fraction captured")
ax.set_title("F1: cross-source subspace energy"); ax.legend(fontsize=8); save(fig, "fig_anatomy_F1_residual.pdf")

# F2 code entropy / effdim vs R@10
g = load("anatomy_F2_lang_geometry.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, key, lab in [(axes[0], "code_bit_entropy", "code bit-entropy (sum, /1024)"),
                     (axes[1], "txtemb_effdim", "text-embedding effective dim")]:
    for r in g:
        c = "#d95f02" if r["script"] == "non-latin" else "#1b9e77"
        x, y = float(r[key]), float(r["r10"]); ax.scatter(x, y, color=c, s=28)
        ax.annotate(r["lang"], (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(lab); ax.set_ylabel("R@10")
axes[0].scatter([], [], color="#1b9e77", label="latin"); axes[0].scatter([], [], color="#d95f02", label="non-latin"); axes[0].legend(fontsize=8)
fig.suptitle("F2: per-language code degeneracy / embedding rank vs R@10 (XM3600)")
save(fig, "fig_anatomy_F2_lang_geometry.pdf")
print("ANATOMY_PLOT_F_DONE", flush=True)
