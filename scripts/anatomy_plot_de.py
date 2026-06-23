"""Anatomy plotting D/E — paper/anatomy_D_*.csv, anatomy_E_*.csv + anatomy_de_arrays.npz -> PDFs."""
from __future__ import annotations
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.environ.get("REPO", "."), "paper")
COL = {"ceiling": "#1b9e77", "distillation": "#d95f02", "head-adapt": "#7570b3"}


def load(n):
    with open(os.path.join(OUT, n)) as f:
        return list(csv.DictReader(f))


def save(fig, n):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, n)); plt.close(fig); print(f"  wrote paper/{n}", flush=True)


# ---------- D1 per-bit flip (vs ceiling) ----------
fl = defaultdict(list)
for r in load("anatomy_D_bitflip.csv"):
    fl[r["path"]].append(float(r["flip_frac"]))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for p in fl:
    ax.hist(fl[p], bins=50, histtype="step", color=COL[p], label=f"{p} (mean {np.mean(fl[p]):.3f})")
ax.set_xlabel("per-bit flip fraction vs ceiling true-code"); ax.set_ylabel("# bits")
ax.set_title("D1: systematic per-bit sign flips relative to ceiling"); ax.legend(fontsize=8)
save(fig, "fig_anatomy_D1_bitflip.pdf")

# ---------- D2 success vs failure features ----------
ff = load("anatomy_D_success_features.csv")
feats = ["pair_hamming", "mean_absz", "caplen", "pair_cosine"]
paths = ["ceiling", "distillation", "head-adapt"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, ft in zip(axes, feats):
    sm, fm, xs = [], [], []
    for p in paths:
        row = next((r for r in ff if r["path"] == p and r["feature"] == ft), None)
        if row and row["success_mean"] not in ("", "None", None):
            xs.append(p); sm.append(float(row["success_mean"])); fm.append(float(row["fail_mean"]))
    x = np.arange(len(xs)); w = 0.35
    ax.bar(x - w/2, sm, w, label="success", color="#4daf4a")
    ax.bar(x + w/2, fm, w, label="fail", color="#e41a1c")
    ax.set_xticks(x); ax.set_xticklabels(xs, rotation=20, fontsize=7); ax.set_title(ft); ax.legend(fontsize=7)
fig.suptitle("D2: success vs failure query means (R@10 top-10 hit)")
save(fig, "fig_anatomy_D2_success_features.pdf")

# ---------- E1 per-language pair-Hamming vs R@10 ----------
lh = load("anatomy_E_lang_hamming.csv")
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for r in lh:
    c = "#d95f02" if r["script"] == "non-latin" else "#1b9e77"
    x, y = float(r["pair_hamming_med"]), float(r["r10"])
    ax.scatter(x, y, color=c, s=30)
    ax.annotate(r["lang"], (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
ax.scatter([], [], color="#1b9e77", label="latin"); ax.scatter([], [], color="#d95f02", label="non-latin")
ax.set_xlabel("median paired text-image Hamming (1024b)"); ax.set_ylabel("R@10")
ax.set_title("E1: per-language pair-Hamming vs R@10 (XM3600, ceiling path)"); ax.legend(fontsize=9)
save(fig, "fig_anatomy_E1_lang_hamming_vs_r10.pdf")

# ---------- E2 language scatter (UMAP of text embeddings) ----------
try:
    import umap
    arr = np.load(os.path.join(OUT, "anatomy_de_arrays.npz"), allow_pickle=True)
    samp_langs = [str(x) for x in arr["samp_langs"]]
    Xs, labs = [], []
    for l in samp_langs:
        e = arr[f"xtxt_{l}"][:400]  # cap per lang for speed
        Xs.append(e); labs += [l]*e.shape[0]
    X = np.concatenate(Xs, 0).astype(np.float32); labs = np.array(labs)
    u = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, l in enumerate(samp_langs):
        m = labs == l; ax.scatter(u[m, 0], u[m, 1], s=6, alpha=0.5, color=cmap(i % 10), label=l)
    ax.set_title("E2: UMAP of per-language text embeddings (XM3600 SigLIP-text)"); ax.legend(fontsize=8, ncol=2)
    save(fig, "fig_anatomy_E2_lang_umap.pdf")
except Exception as e:
    print("E2 skipped:", e, flush=True)

# ---------- E3 int8 flip per bit ----------
i8 = [float(r["flip_frac"]) for r in load("anatomy_E_int8_flip.csv")]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(len(i8)), i8, lw=0.5, color="#377eb8")
ax.set_xlabel("bit index (Matryoshka order)"); ax.set_ylabel("int8 sign-flip fraction")
ax.set_title(f"E3: int8-quantization bit-flip sensitivity (mean {np.mean(i8):.4f})")
save(fig, "fig_anatomy_E3_int8_flip.pdf")

print("ANATOMY_PLOT_DE_DONE", flush=True)
