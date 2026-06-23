"""Plot D-deep: flip vs |z| decile, separation gap by success/fail."""
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


# D5 flip vs |z| decile
fv = defaultdict(list)
for r in load("anatomy_Ddeep_flip_vs_absz.csv"):
    fv[r["path"]].append((int(r["absz_decile"]), float(r["flip_frac"])))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for p in fv:
    d, f = zip(*sorted(fv[p])); ax.plot(d, f, "o-", color=COL[p], label=p)
ax.set_xlabel("|z| decile (1=smallest margin → 10=largest)"); ax.set_ylabel("flip fraction vs ceiling true-code")
ax.set_title("D5: per-bit flip vs pre-sign |z| (boundary vs confident flips)"); ax.legend(fontsize=8)
save(fig, "fig_anatomy_D5_flip_vs_absz.pdf")

# D6 separation gap by success/fail
gp = defaultdict(lambda: defaultdict(list))
for r in load("anatomy_Ddeep_gap.csv"):
    gp[r["path"]][r["group"]].append((int(r["gap_bin"]), int(r["count"])))
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
for ax, p in zip(axes, ["ceiling", "distillation", "head-adapt"]):
    for grp, c in [("success", "#4daf4a"), ("fail", "#e41a1c")]:
        if gp[p][grp]:
            x, n = zip(*sorted(gp[p][grp])); ax.plot(x, n, color=c, label=grp, drawstyle="steps-mid")
    ax.axvline(0, color="k", lw=0.5, ls=":"); ax.set_title(p); ax.set_xlabel("gap = nearest_wrong − pair (bits)"); ax.legend(fontsize=8)
axes[0].set_ylabel("# queries")
fig.suptitle("D6: separation gap distribution, success vs failure queries (1024b)")
save(fig, "fig_anatomy_D6_gap_success_fail.pdf")
print("ANATOMY_PLOT_DDEEP_DONE", flush=True)
