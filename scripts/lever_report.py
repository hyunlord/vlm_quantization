"""Analyze /tmp/lever_all.csv (fetched from DGX): average seeds, compute
Delta-vs-baseline per bit, and print gate verdicts. Baseline = bn-infonce@15
(current head, full recipe). The bn-infonce@25 row is a fairness validation
(compared to @15 separately). hmargin margin variants and epochs are part of the key.

Usage: python scripts/lever_report.py [csv]  (default paper/lever_all.csv)
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "paper/lever_all.csv"
BITS = [64, 128, 256, 512, 1024]
NUM = ["en_t2i_r1", "en_t2i_r5", "en_t2i_r10", "en_i2t_r10",
       "ko_t2i_r1", "ko_t2i_r5", "ko_t2i_r10", "hmargin_bits", "hmargin_std", "naninf"]

rows = list(csv.DictReader(open(SRC)))
# config key = everything that defines a run except seed and bit
def ckey(r):
    return (r["head_norm"], r["loss"], r["negsep_w"], r["hmargin_m"],
            r["aux_scale"], r.get("epochs", "15"))

agg, seeds = defaultdict(lambda: defaultdict(float)), defaultdict(set)
seed_r10 = defaultdict(list)
for r in rows:
    k = ckey(r) + (int(r["bit"]),)
    seeds[k].add(r["seed"])
    for c in NUM:
        agg[k][c] += float(r[c])
    seed_r10[k].append(float(r["en_t2i_r10"]))
mean = {k: {c: agg[k][c] / len(seeds[k]) for c in NUM} for k in agg}


def label(c):
    h, l, nw, hm, aux, ep = c
    s = f"{h}/{l}"
    if l in ("hmargin", "infonce_hmargin"):
        s += f"(m{hm})"
    if l in ("infonce_hardneg", "infonce_hmargin"):
        s += f"(w{nw})"
    if float(aux) != 1.0:
        s += f"(aux{aux})"
    if ep != "15":
        s += f"@{ep}ep"
    return s


BASE = ("bn", "infonce", "0.5", "0.1", "1.0", "15")  # baseline config key (sans bit)
def base(bit):
    return mean[BASE + (bit,)]


def configs():
    out = []
    for k in mean:
        c = k[:-1]
        if c not in out:
            out.append(c)
    # baseline first, then by head/loss
    out.sort(key=lambda c: (c != BASE, c))
    return out


print(f"source: {SRC}  ({len(rows)} rows, {len(set(r['seed'] for r in rows))} seeds)\n")
hdr = f"{'config':30} " + " ".join(f"{b:>5}b" for b in BITS) + "   | dR@10 vs bn-infonce@15"
print("EN T2I R@10 (seed-mean) + Delta vs baseline (bn-infonce@15)")
print(hdr); print("-" * len(hdr))
for c in configs():
    line = f"{label(c):30} "
    deltas = []
    for b in BITS:
        k = c + (b,)
        if k not in mean:
            line += f"{'--':>6}"; deltas.append(None); continue
        v = mean[k]["en_t2i_r10"]
        line += f"{v:6.2f}"
        deltas.append(v - base(b)["en_t2i_r10"] if BASE + (b,) in mean else None)
    ds = " ".join(f"{d:+.2f}" if d is not None else "  .  " for d in deltas)
    print(line + "   | " + ds)

print("\nKO T2I R@10 (seed-mean)")
print(hdr); print("-" * len(hdr))
for c in configs():
    line = f"{label(c):30} "
    for b in BITS:
        k = c + (b,)
        line += f"{mean[k]['ko_t2i_r10']:6.2f}" if k in mean else f"{'--':>6}"
    print(line)

print("\nRealized test hmargin (bits) @1024 — Lever A mechanism check")
for c in configs():
    k = c + (1024,)
    if k in mean:
        print(f"  {label(c):30} hmargin {mean[k]['hmargin_bits']:+8.2f} ± {mean[k]['hmargin_std']:6.2f}  "
              f"naninf {mean[k]['naninf']:.0f}")

print("\nseed noise (max-min EN T2I R@10) @1024b:")
for c in configs():
    vs = seed_r10.get(c + (1024,), [])
    print(f"  {label(c):30} {max(vs)-min(vs):.2f}  (n={len(vs)})" if len(vs) > 1 else f"  {label(c):30} (1 seed)")

# ---- fairness validation ----
v25 = ("bn", "infonce", "0.5", "0.1", "1.0", "25")
if v25 + (1024,) in mean:
    print("\nFairness validation (baseline convergence):")
    for b in BITS:
        if v25 + (b,) in mean and BASE + (b,) in mean:
            d = mean[v25 + (b,)]["en_t2i_r10"] - mean[BASE + (b,)]["en_t2i_r10"]
            print(f"  {b}b: bn-infonce@25 {mean[v25+(b,)]['en_t2i_r10']:.2f} vs @15 "
                  f"{mean[BASE+(b,)]['en_t2i_r10']:.2f}  (d {d:+.2f})")
    print("  -> if |d|>0.5pt, 15 epochs under-trains the baseline; re-run gate at 25.")

# ---- gate ----
print("\n=== GATE (EN T2I R@10, dR@10 vs bn-infonce@15) ===")
def best_delta(c):
    best, bb, cons = -99.0, None, 0
    for b in BITS:
        k = c + (b,)
        if k in mean and BASE + (b,) in mean:
            d = mean[k]["en_t2i_r10"] - base(b)["en_t2i_r10"]
            if d > best:
                best, bb = d, b
            if d >= 1.0:
                cons += 1
    return best, bb, cons

for c in configs():
    if c == BASE:
        continue
    bd, bb, cons = best_delta(c)
    green = bd >= 1.0 and cons >= 3
    print(f"  {label(c):30} best {bd:+.2f}pt @{bb}b  bits>=+1.0: {cons}  -> {'GREEN' if green else 'red'}")

# composition-response arm (if present)
print("\nComposition-response (aux weight effect) @1024b EN T2I:")
for h in ("bn", "none"):
    vals = {}
    for aux in ("0.0", "1.0", "4.0"):
        k = (h, "infonce", "0.5", "0.1", aux, "15", 1024)
        if k in mean:
            vals[aux] = mean[k]["en_t2i_r10"]
    if len(vals) >= 2:
        sp = max(vals.values()) - min(vals.values())
        print(f"  {h}: " + " ".join(f"aux{a}={vals[a]:.2f}" for a in sorted(vals)) +
              f"  spread {sp:.2f}pt ({'RESPONDS>=0.5' if sp >= 0.5 else 'inert'})")
