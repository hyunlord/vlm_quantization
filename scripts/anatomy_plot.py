"""Anatomy plotting (A/B/C) — read paper/anatomy_*.csv + anatomy_arrays.npz -> paper/fig_anatomy_*.pdf.
Neutral figures only. Run: .venv/bin/python scripts/anatomy_plot.py
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.environ.get("REPO", "."), "paper")
PATHS = ["image", "ceiling", "distillation", "head-adapt"]
COL = {"image": "#444444", "ceiling": "#1b9e77", "distillation": "#d95f02", "head-adapt": "#7570b3"}


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name)); plt.close(fig)
    print(f"  wrote paper/{name}", flush=True)


arr = np.load(os.path.join(OUT, "anatomy_arrays.npz"), allow_pickle=True)

# ---------- A1 SV spectrum ----------
sv = load("anatomy_A_svspectrum.csv")
pr = {r["source"]: float(r["singular_value"]) for r in sv if r["rank"] == "-1"}
fig, ax = plt.subplots(figsize=(6, 4.2))
for s in PATHS:
    pts = [(int(r["rank"]), float(r["var_frac"])) for r in sv if r["source"] == s and r["rank"] != "-1"]
    pts = [p for p in pts if p[1] > 0]
    if pts:
        x, y = zip(*pts); ax.semilogy(x, y, color=COL[s], label=f"{s} (PR={pr.get(s,0):.0f})")
ax.set_xlabel("singular-value rank"); ax.set_ylabel("variance fraction"); ax.set_title("A1: backbone embedding singular-value spectrum")
ax.legend(fontsize=8); save(fig, "fig_anatomy_A1_svspectrum.pdf")

# ---------- A2 per-dim variance (sorted) ----------
pv = defaultdict(list)
for r in load("anatomy_A_perdim_var.csv"):
    pv[r["source"]].append(float(r["var"]))
fig, ax = plt.subplots(figsize=(6, 4.2))
for s in PATHS:
    v = np.sort(np.array(pv[s]))[::-1]; ax.semilogy(np.arange(len(v)), v + 1e-12, color=COL[s], label=s)
ax.set_xlabel("dim (sorted by variance)"); ax.set_ylabel("per-dim variance"); ax.set_title("A2: per-dimension variance (sorted)")
ax.legend(fontsize=8); save(fig, "fig_anatomy_A2_perdim_var.pdf")

# ---------- A3 norm hist ----------
nh = defaultdict(list)
for r in load("anatomy_A_norm_hist.csv"):
    nh[r["source"]].append((float(r["norm_bin"]), int(r["count"])))
fig, ax = plt.subplots(figsize=(6, 4.2))
for s in PATHS:
    if nh[s]:
        x, c = zip(*nh[s]); ax.plot(x, c, color=COL[s], label=s, drawstyle="steps-mid")
ax.set_xlabel("raw embedding L2 norm (pre-normalization)"); ax.set_ylabel("count"); ax.set_title("A3: raw embedding norm distribution")
ax.legend(fontsize=8); save(fig, "fig_anatomy_A3_norm_hist.pdf")

# ---------- A4 cosine pair vs random ----------
cz = defaultdict(lambda: defaultdict(list))
for r in load("anatomy_A_cosine_pair_vs_random.csv"):
    cz[r["path"]][r["kind"]].append((float(r["cos_bin"]), int(r["count"])))
fig, ax = plt.subplots(figsize=(6, 4.2))
for p in cz:
    for kind, ls in [("pair", "-"), ("random", "--")]:
        if cz[p][kind]:
            x, c = zip(*cz[p][kind]); ax.plot(x, c, ls, color=COL.get(p, "#000"), label=f"{p} {kind}", drawstyle="steps-mid")
ax.set_xlabel("cosine(text, image)"); ax.set_ylabel("count"); ax.set_title("A4: paired vs random cross-modal cosine (SigLIP space)")
ax.legend(fontsize=8); save(fig, "fig_anatomy_A4_cosine.pdf")

# ---------- A5 PCA + UMAP scatter (1152-space sources) ----------
from sklearn.decomposition import PCA
try:
    import umap
    have_umap = True
except Exception:
    have_umap = False
S1152 = ["image", "ceiling", "distillation"]  # shared SigLIP space
X = np.concatenate([arr[f"emb_{s}"] for s in S1152], 0)
lab = np.concatenate([[i]*arr[f"emb_{s}"].shape[0] for i, s in enumerate(S1152)])
p2 = PCA(n_components=2).fit_transform(X)
fig, axes = plt.subplots(1, 2 if have_umap else 1, figsize=(11 if have_umap else 6, 4.6), squeeze=False)
for i, s in enumerate(S1152):
    m = lab == i; axes[0][0].scatter(p2[m, 0], p2[m, 1], s=4, alpha=0.4, color=COL[s], label=s)
axes[0][0].set_title("A5: PCA-2D (image/ceiling/distillation)"); axes[0][0].legend(fontsize=8)
if have_umap:
    u2 = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X)
    for i, s in enumerate(S1152):
        m = lab == i; axes[0][1].scatter(u2[m, 0], u2[m, 1], s=4, alpha=0.4, color=COL[s], label=s)
    axes[0][1].set_title("A5: UMAP-2D (image/ceiling/distillation)"); axes[0][1].legend(fontsize=8)
save(fig, "fig_anatomy_A5_scatter.pdf")

# ---------- B1 stage per-dim variance (violin per stage/path) ----------
sv2 = defaultdict(lambda: defaultdict(list))
for r in load("anatomy_B_stage_perdim.csv"):
    sv2[r["path"]][r["stage"]].append(float(r["var"]))
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
for ax, p in zip(axes, PATHS):
    data = [sv2[p][st] for st in ("sliced", "bn", "z")]
    ax.violinplot(data, showmeans=True); ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["sliced", "bn", "z"])
    ax.set_yscale("log"); ax.set_title(p); ax.set_ylabel("per-dim variance" if p == "image" else "")
fig.suptitle("B1: per-dim variance by head stage (sliced -> BN -> L2 z)")
save(fig, "fig_anatomy_B1_stage_var.pdf")

# ---------- B2 stage value histograms ----------
sh = defaultdict(lambda: defaultdict(list))
for r in load("anatomy_B_stage_hist.csv"):
    sh[r["path"]][r["stage"]].append((float(r["val_bin"]), int(r["count"])))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, st in zip(axes, ("sliced", "bn", "z")):
    for p in PATHS:
        if sh[p][st]:
            x, c = zip(*sh[p][st]); ax.plot(x, c, color=COL[p], label=p, drawstyle="steps-mid")
    ax.set_title(f"stage: {st}"); ax.set_xlabel("value"); ax.legend(fontsize=7)
fig.suptitle("B2: value distribution per head stage")
save(fig, "fig_anatomy_B2_stage_hist.pdf")

# ---------- B3 |z| distribution ----------
az = defaultdict(list)
for r in load("anatomy_B_absz.csv"):
    if not str(r["absz_bin"]).startswith("frac"):
        az[r["path"]].append((float(r["absz_bin"]), int(r["count"])))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for p in PATHS:
    if az[p]:
        x, c = zip(*az[p]); axes[0].plot(x, c, color=COL[p], label=p, drawstyle="steps-mid")
        axes[1].plot(x, c, color=COL[p], label=p, drawstyle="steps-mid")
axes[0].set_title("B3: |z| (pre-sign) distribution"); axes[0].set_xlabel("|z|"); axes[0].legend(fontsize=8)
axes[1].set_xlim(0, 0.02); axes[1].set_title("B3: |z| zoom near 0 (sign-flip boundary)"); axes[1].set_xlabel("|z|")
save(fig, "fig_anatomy_B3_absz.pdf")

# ---------- C1 bit balance ----------
bb = defaultdict(list)
for r in load("anatomy_C_bit_balance.csv"):
    bb[r["path"]].append(float(r["frac_pos"]))
fig, ax = plt.subplots(figsize=(6, 4.2))
for p in PATHS:
    ax.hist(bb[p], bins=50, range=(0, 1), histtype="step", color=COL[p], label=p)
ax.axvline(0.5, color="k", lw=0.5, ls=":"); ax.set_xlabel("per-bit +1 fraction"); ax.set_ylabel("# bits")
ax.set_title("C1: per-bit activation balance (1024 bits)"); ax.legend(fontsize=8); save(fig, "fig_anatomy_C1_bitbalance.pdf")

# ---------- C2 bit-bit correlation heatmaps ----------
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, p in zip(axes, PATHS):
    cm = arr[f"corr_{p}"].astype(np.float32)
    im = ax.imshow(cm[:256, :256], cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_title(f"{p}"); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.7); fig.suptitle("C2: bit-bit correlation (first 256 bits)")
save(fig, "fig_anatomy_C2_bitcorr.pdf")

# ---------- C3 Hamming distributions ----------
hd = defaultdict(lambda: defaultdict(list))
for r in load("anatomy_C_hamming_dist.csv"):
    hd[r["path"]][r["kind"]].append((float(r["ham_bin"]), int(r["count"])))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for p in ["ceiling", "distillation", "head-adapt"]:
    for kind, ls in [("pair", "-"), ("nearest_wrong", "--")]:
        if hd[p][kind]:
            x, c = zip(*hd[p][kind]); axes[0].plot(x, c, ls, color=COL[p], label=f"{p} {kind}", drawstyle="steps-mid")
    if hd[p]["gap(nwrong-pair)"]:
        x, c = zip(*hd[p]["gap(nwrong-pair)"]); axes[1].plot(x, c, color=COL[p], label=p, drawstyle="steps-mid")
axes[0].set_title("C3: pair vs nearest-wrong Hamming (1024b)"); axes[0].set_xlabel("Hamming distance"); axes[0].legend(fontsize=7)
axes[1].axvline(0, color="k", lw=0.5, ls=":"); axes[1].set_title("C3: separation gap (nearest_wrong - pair)"); axes[1].set_xlabel("gap (bits)"); axes[1].legend(fontsize=8)
save(fig, "fig_anatomy_C3_hamming.pdf")

# ---------- C4 bit ablation grouped ----------
ab = defaultdict(list)
for r in load("anatomy_C_bit_ablation_grouped.csv"):
    ab[r["path"]].append((int(r["group"]), float(r["r10_drop"])))
fig, ax = plt.subplots(figsize=(7, 4.2))
for p in ["ceiling", "distillation", "head-adapt"]:
    g, d = zip(*sorted(ab[p])); ax.plot(g, d, "o-", color=COL[p], label=p)
ax.set_xlabel("bit group (each = 64 contiguous bits)"); ax.set_ylabel("R@10 drop when group removed")
ax.set_title("C4: grouped bit-ablation contribution"); ax.legend(fontsize=8); save(fig, "fig_anatomy_C4_ablation.pdf")

print("ANATOMY_PLOT_ABC_DONE", flush=True)
