"""Analyze /tmp/lever_all.csv (fetched from DGX): average seeds, compute
Delta-vs-baseline per bit, split into paper/lever_A.csv + paper/lever_B.csv,
and print gate verdicts. Baseline = bn-infonce (current head).

Usage: python scripts/lever_report.py [path_to_lever_all.csv]
"""
from __future__ import annotations
import csv, sys, os
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "paper/lever_all.csv"
BITS = [64, 128, 256, 512, 1024]
NUM = ["en_t2i_r1", "en_t2i_r5", "en_t2i_r10", "en_i2t_r10",
       "ko_t2i_r1", "ko_t2i_r5", "ko_t2i_r10", "hmargin_bits", "hmargin_std", "naninf"]

rows = list(csv.DictReader(open(SRC)))
# group by (head_norm, loss, negsep_w, aux_scale, bit) averaged over seeds
key = lambda r: (r["head_norm"], r["loss"], r["negsep_w"], r["aux_scale"], int(r["bit"]))
agg, seeds = defaultdict(lambda: defaultdict(float)), defaultdict(set)
for r in rows:
    k = key(r); seeds[k].add(r["seed"])
    for c in NUM:
        agg[k][c] += float(r[c])
mean = {k: {c: agg[k][c] / len(seeds[k]) for c in NUM} for k in agg}
seed_vals = defaultdict(lambda: defaultdict(list))
for r in rows:
    seed_vals[key(r)]["en_t2i_r10"].append(float(r["en_t2i_r10"]))


def cfg_label(head, loss, nw, aux):
    s = f"{head}/{loss}"
    if loss in ("infonce_hardneg", "infonce_hmargin") and float(nw) != 0.5:
        s += f"(w{nw})"
    if float(aux) != 1.0:
        s += f"(aux{aux})"
    return s


def base(bit):
    return mean[("bn", "infonce", "0.5", "1.0", bit)]


def configs():
    seen = []
    for (h, l, nw, aux, b) in mean:
        c = (h, l, nw, aux)
        if c not in seen:
            seen.append(c)
    return seen


def seed_noise(c, bit):
    vs = seed_vals[(c[0], c[1], c[2], c[3], bit)].get("en_t2i_r10", [])
    return (max(vs) - min(vs)) if len(vs) > 1 else 0.0


print(f"source: {SRC}  ({len(rows)} rows)\n")
hdr = f"{'config':28} " + " ".join(f"{b:>5}b" for b in BITS) + "   | dR@10 vs bn-infonce"
print("EN T2I R@10 (seed-mean) + Delta vs baseline")
print(hdr); print("-" * len(hdr))
for c in configs():
    line = f"{cfg_label(*c):28} "
    deltas = []
    for b in BITS:
        k = (c[0], c[1], c[2], c[3], b)
        if k not in mean:
            line += f"{'--':>6}"; deltas.append(None); continue
        v = mean[k]["en_t2i_r10"]; d = v - base(b)["en_t2i_r10"]
        line += f"{v:6.2f}"; deltas.append(d)
    ds = " ".join(f"{d:+.2f}" if d is not None else "  -- " for d in deltas)
    line += f"   | {ds}"
    print(line)

print("\nKO T2I R@10 (seed-mean)")
print(hdr); print("-" * len(hdr))
for c in configs():
    line = f"{cfg_label(*c):28} "
    for b in BITS:
        k = (c[0], c[1], c[2], c[3], b)
        line += f"{mean[k]['ko_t2i_r10']:6.2f}" if k in mean else f"{'--':>6}"
    print(line)

print("\nRealized test hmargin (bits) @1024 — mechanism check for Lever A")
for c in configs():
    k = (c[0], c[1], c[2], c[3], 1024)
    if k in mean:
        print(f"  {cfg_label(*c):28} hmargin {mean[k]['hmargin_bits']:+7.2f} ± {mean[k]['hmargin_std']:.2f}  "
              f"naninf {mean[k]['naninf']:.0f}")

print("\nseed noise (max-min EN T2I R@10) at 1024b:")
for c in configs():
    print(f"  {cfg_label(*c):28} {seed_noise(c, 1024):.2f}")

# ---- gate verdicts ----
print("\n=== GATE ===")
LEVER_A = [("bn", "hardneg"), ("bn", "hmargin"), ("bn", "infonce_hardneg"), ("bn", "infonce_hmargin")]
LEVER_B = [("ln", "infonce"), ("none", "infonce"), ("none_scale", "infonce"),
           ("rotation", "infonce"), ("none", "infonce_hmargin")]


def best_delta(h, l):
    best, bestbit = -99, None
    consistent = 0
    for b in BITS:
        k = (h, l, "0.5", "1.0", b)
        if k in mean:
            d = mean[k]["en_t2i_r10"] - base(b)["en_t2i_r10"]
            if d > best:
                best, bestbit = d, b
            if d >= 1.0:
                consistent += 1
    return best, bestbit, consistent


for name, lst in [("Lever A", LEVER_A), ("Lever B", LEVER_B)]:
    print(f"\n{name}: best dR@10 per variant (consistency = # bits with dR@10>=+1.0)")
    for (h, l) in lst:
        bd, bb, cons = best_delta(h, l)
        verdict = "GREEN" if (bd >= 1.0 and cons >= 3) else "red"
        print(f"  {h}/{l:18} best {bd:+.2f}pt @{bb}b  bits>=+1.0: {cons}  -> {verdict}")

# composition-response arm (Lever B analysis)
print("\nComposition-response arm (does aux weight move R@10?) @1024b EN T2I:")
for h in ("bn", "none"):
    vals = {}
    for aux in ("0.0", "1.0", "4.0"):
        k = (h, "infonce", "0.5", aux, 1024)
        if k in mean:
            vals[aux] = mean[k]["en_t2i_r10"]
    if len(vals) >= 2:
        spread = max(vals.values()) - min(vals.values())
        print(f"  {h}: aux0={vals.get('0.0','--')} aux1={vals.get('1.0','--')} "
              f"aux4={vals.get('4.0','--')}  spread {spread:.2f}pt "
              f"({'RESPONDS >=0.5' if spread >= 0.5 else 'inert'})")
