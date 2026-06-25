"""Equal-byte Pareto figure (PDF) from paper/equalbyte_pareto.csv. Accuracy (R@10) vs bytes/image,
all methods; binary_head highlighted to show it is on/under the Pareto front. No data distortion."""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); PAP = os.path.join(HERE, "..", "paper")
OUT = os.path.join(PAP, "equalbyte_figs"); os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(PAP, "equalbyte_pareto.csv"))))
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})
B = [16, 32, 64, 128, 256]
style = {"binary_head": ("#d62728", "o", "binary Hamming (ours)"),
         "asym_binary": ("#9467bd", "D", "asymmetric binary"),
         "int8_headz": ("#2ca02c", "s", "int8 head-z"),
         "fp16_headz": ("#1f77b4", "^", "fp16 head-z"),
         "int8_pca": ("#8c564b", "v", "int8 raw-PCA"),
         "fp16_pca": ("#7f7f7f", "x", "fp16 raw-PCA"),
         "pq": ("#ff7f0e", "P", "PQ (faiss)")}
def series(m, col):
    out = {}
    for r in rows:
        if r["method"] == m and r[col] not in ("", "None"):
            out[int(r["bytes"])] = float(r[col])
    return out
for lang, col in [("EN", "r10_en"), ("KO", "r10_ko")]:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for m, (c, mk, lab) in style.items():
        d = series(m, col)
        if not d: continue
        xs = sorted(d); ys = [d[x] for x in xs]
        lw = 2.4 if m == "binary_head" else 1.3
        ax.plot(xs, ys, marker=mk, color=c, lw=lw, ms=5 if m == "binary_head" else 4, label=lab,
                zorder=5 if m == "binary_head" else 3)
    ax.set_xscale("log", base=2); ax.set_xticks(B); ax.set_xticklabels([f"{b}" for b in B])
    ax.set_xlabel("bytes / image"); ax.set_ylabel(f"COCO {lang} R@10")
    ax.legend(fontsize=6.2, loc="lower right", ncol=2)
    ax.set_title(f"Equal-byte accuracy ({lang})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"pareto_{lang}.pdf")); plt.close(fig)
print("wrote", os.listdir(OUT))
