#!/usr/bin/env python3
"""Generate paper figures (Fig 2-5) from clone-verified repo CSVs -> paper/iaai/figs/*.pdf.
Fig 1 (system diagram) is TikZ inline in main.tex. Every figure's numbers come ONLY from the
named CSV; no axis truncation that would distort. Run: .venv/bin/python paper/iaai/make_figs.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAP = os.path.normpath(os.path.join(HERE, ".."))          # paper/
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.linewidth": 0.6,
                     "legend.fontsize": 7, "figure.dpi": 200})
COL = 3.3   # AAAI single-column inches
DBL = 7.0   # AAAI double-column inches


def read(name):
    with open(os.path.join(PAP, name)) as f:
        return list(csv.DictReader(f))


# ---- Fig 2: bit-rate sweep (bits_extreme_lc.csv) ----
def fig_bits():
    rows = read("bits_extreme_lc.csv")
    def series(lang, pre):
        d = [(int(r["bits"]), float(r["R10"])) for r in rows
             if r["dataset"] == "coco" and r["lang"] == lang and r["preproc"] == pre]
        d.sort(); return [b for b, _ in d], [v for _, v in d]
    bx, by = series("en", "lower")
    kx, ky = series("ko", "orig(caseless)")
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(bx, by, "o-", lw=1.6, ms=4, color="#1f77b4", label="English")
    ax.plot(kx, ky, "s--", lw=1.4, ms=3.5, color="#d62728", label="Korean (caseless)")
    ax.set_xscale("log", base=2); ax.set_xticks(bx)
    ax.set_xticklabels([str(b) for b in bx], rotation=45, fontsize=6)
    ax.axvline(1024, color="gray", ls=":", lw=1)
    ax.annotate("1024-bit\n(128 B)\nsweet spot", xy=(1024, by[bx.index(1024)]),
                xytext=(150, 30), fontsize=6.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"))
    ax.set_xlabel("code length (bits)"); ax.set_ylabel("R@10")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_bits.pdf")); plt.close(fig)


# ---- Fig 3: sign information loss / asymmetric (signloss_quant.csv, ceiling path) ----
def fig_signloss():
    rows = [r for r in read("signloss_quant.csv") if r["path"] == "ceiling"]
    rows.sort(key=lambda r: int(r["bit"]))
    b = [int(r["bit"]) for r in rows]
    cont = [float(r["r10_continuous"]) for r in rows]
    asym = [float(r["r10_asym"]) for r in rows]
    ham = [float(r["r10_hamming"]) for r in rows]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(b, cont, "o-", lw=1.6, ms=4, color="#2ca02c", label="continuous (ceiling)")
    ax.plot(b, asym, "^-", lw=1.5, ms=4, color="#ff7f0e", label="asymmetric")
    ax.plot(b, ham, "s-", lw=1.5, ms=4, color="#d62728", label="Hamming (1-bit)")
    ax.fill_between(b, ham, cont, color="#d62728", alpha=0.08)
    ax.annotate("sign loss\n+13.9 @64-bit", xy=(64, (cont[0] + ham[0]) / 2),
                xytext=(180, 60), fontsize=6.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"))
    ax.set_xscale("log", base=2); ax.set_xticks(b)
    ax.set_xticklabels([str(x) for x in b], fontsize=6.5)
    ax.set_xlabel("code length (bits)"); ax.set_ylabel("COCO R@10")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_signloss.pdf")); plt.close(fig)


# ---- Fig 4: multilingual 36-lang (multiling_server_lc.csv) ----
def fig_multiling():
    rows = read("multiling_server_lc.csv")
    def f(x):
        try: return float(x)
        except: return None
    data = []
    for r in rows:
        s, c, o = f(r["server_lc_R10"]), f(r["ceiling_lc"]), f(r["offline_best"])
        if s is None: continue
        data.append((r["lang"], s, c, o))
    data.sort(key=lambda t: t[1], reverse=True)
    langs = [d[0] for d in data]; srv = [d[1] for d in data]
    ceil = [d[2] for d in data]; off = [d[3] for d in data]
    x = range(len(langs))
    fig, ax = plt.subplots(figsize=(DBL, 2.5))
    ax.plot(x, ceil, ".", ms=5, color="#7f7f7f", label="float ceiling")
    ax.plot(x, srv, "o-", lw=1.2, ms=3, color="#1f77b4", label="server 1-bit (ours)")
    ax.plot(x, off, "s", ms=3.5, color="#ff7f0e", label="offline MiniLM (browser)")
    # mark offline > server
    win = {"hi", "te", "th", "mi"}
    for i, lg in enumerate(langs):
        if lg in win:
            ax.annotate(lg, (i, off[i]), fontsize=6, color="#ff7f0e",
                        xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_xticks(list(x)); ax.set_xticklabels(langs, rotation=90, fontsize=5.5)
    ax.set_ylabel("XM3600 R@10"); ax.set_xlabel("language (sorted by server R@10)")
    ax.legend(loc="upper right", ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_multiling.pdf")); plt.close(fig)


# ---- Fig 5: 11-gate negative-space summary (verdicts from STATE_OF_PROJECT, clone-verified) ----
def fig_gates():
    # (label, verdict, delta-or-metric). Verdicts match STATE_OF_PROJECT gate table + appendix.
    gates = [
        ("cosine predictor", "RED", "rho=0.79"),
        ("parity predictor", "RED", "anti-corr"),
        ("margin predictor", "RED", "pinned"),
        ("neighbor predictor", "RED", "flat"),
        ("diralign predictor", "RED", "flat"),
        ("rank / eff-dim", "RED", "rho=0.12/0.38"),
        ("Lever A neg-sep loss", "RED", "R@1+/R@10-"),
        ("Lever B BN-free head", "inert", "<=0.2pt"),
        ("residual coding", "inert", "-1.6pt"),
        ("multiscale/UNet head", "RED", "-8.9pt"),
        ("Lever D dynamic bits", "GREEN", "system win"),
    ]
    color = {"RED": "#d62728", "inert": "#bbbbbb", "GREEN": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    y = range(len(gates))[::-1]
    for yi, (lab, v, m) in zip(y, gates):
        ax.barh(yi, 1, color=color[v], alpha=0.85, height=0.7)
        ax.text(0.03, yi, lab, va="center", ha="left", fontsize=6.5, color="white")
        ax.text(1.02, yi, f"{v} ({m})", va="center", ha="left", fontsize=6)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(gates) - 0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("11-gate negative-space map", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_gates.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_bits(); fig_signloss(); fig_multiling(); fig_gates()
    print("figs written to", FIG, ":", sorted(os.listdir(FIG)))
