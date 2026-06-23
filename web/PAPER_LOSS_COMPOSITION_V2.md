# PAPER_LOSS_COMPOSITION_V2 — loss-composition gate, CODE-DEFINING regime · NEGATIVE

Branch `paper-loss-composition-v2`. Tests the AAAI main claim — *"in binary-code learning the **composition** of
loss terms (weighted-sum vs curriculum vs unified) dominates weight-tuning for accuracy AND robustness"* — in the
regime where composition CAN matter: **no anchor**. Both `img_h` and `txt_h` are trained from scratch on frozen
SigLIP2-So400m features, jointly DEFINING the 1024-bit code space (the original ft113 setup). Companion to v1
(`paper-loss-composition` f0af984), which found W/C/U inert in the anchor-adaptation regime.

**Result: composition is inert here too.** Combined with v1 this is a robust **two-regime BatchNorm-dominance
(simplicity) finding** — exactly the honest negative the work order anticipated.

## Setup (head-only; encoder frozen)
- Train: `/tmp/emb_aug.pt` SigLIP2 train features (img=clean 113,287×1152, txt 113,287×1152), L2-normed.
- Both heads trainable, joint AdamW, 8 epochs, OneCycleLR. Symmetric code-defining loss (math verified):
  - `align = InfoNCE(tanh(z_txt/T), tanh(z_img/T))` (paired, both grad) + `quant = EAQL` on both heads +
    `margin = relu(m−|z|)` on both (squared for U) + `nesting = LCS` on both. m = m_rel·mean_bits(std(z)).
  - **(W)** `align + wq·quant + wm·margin + wl·nesting`, T=1; weight grid wq×wm = 3×3.
  - **(C)** curriculum: T annealed 4→1 (cosine), wq/wm ramp 0→target over first 60%; T0∈{2,4}.
  - **(U)** unified: `align + wu·relu(m−|z|)² + wl·nesting`; grid (wu,m_rel)=6 pts.
  - Best setting of each re-run at a 2nd seed for the noise band.
- Eval: gallery = sign(img_h(test_img)), query = sign(txt_h(test_txt EN)); faiss IndexBinaryFlat at nested bits
  {64,128,256,512,1024}; COCO 5K test. Output `paper/loss_composition_v2.csv` (100 rows, 17 configs+variance, 34 min).
- **Gate epochs = 8** (relative comparison). **EN-primary**: KO deferred to a positive gate (needs fresh
  SigLIP2-text-KO encoding; EN alone decides whether composition matters, and KO would mirror it).

## Result — R@10 EN by bit (seed-42 grid: min..max across each comp's settings)

| bit | W (min..max, **range**) | C | U |
|----:|---|---|---|
| 64   | 65.56..66.12 (**0.56**) | 65.92..66.02 | 66.10 |
| 128  | 72.20..72.58 (**0.38**) | 72.44..72.88 | 72.76..72.78 |
| 256  | 75.30..75.82 (**0.52**) | 75.70..75.74 | 75.58 |
| 512  | 76.86..77.14 (**0.28**) | 76.96..76.98 | 77.38..77.40 |
| 1024 | 77.90..78.14 (**0.24**) | 77.94..78.10 | 77.94..77.98 |

Best-of-each @1024, seeds {42,0}: **W** 78.14/78.26 (mean 78.20), **C** 78.1/**78.9** (mean 78.50), **U** 77.98/78.64
(mean 78.31). **Seed spread: C 0.80, U 0.66, W 0.12.** Diagnostics across all 17 configs @1024: margin 0.0251–0.0252,
cos_pair 0.7729 (spread **0.0000**), flip 0.943–0.952%.

## Verdict: GATE FAILS (negative) — both criteria, honestly
1. **C/U do not beat W beyond seed noise.** Best-C-mean (78.50) − best-W-mean (78.20) = **0.30 pt**, *inside* the
   per-setting seed spread (0.66–0.80 pt). The only points exceeding W's max — C@128 (+0.30), U@512 (+0.26) — do **not**
   hold at other bits (U is bottom-of-pack at 1024), so they are noise, not a consistent cross-bit lever.
2. **No robustness gap to close.** W's weight-sensitivity range (0.24–0.56 pt) is comparable to or smaller than seed
   noise. W is already weight-insensitive — there is nothing for C/U to flatten.

## Mechanism (the honest deliverable)
Diagnostics are **constant to 3–4 digits across all compositions and weights** (cos_pair spread 0.0000). The
margin/quant/curriculum terms do not move the code geometry even when the codes are **learned from scratch**. Cause
is the **per-bit BatchNorm + L2-norm** on the pre-sign vector: they fix the scale/geometry, so InfoNCE alignment
sets the directions and the auxiliary terms have nothing left to shape. This is the SAME mechanism as v1, now shown
in the opposite regime.

## Two-regime finding (v1 + v2 — the contribution)
| regime | setup | abs. R@10@1024 | W/C/U | mechanism |
|---|---|---|---|---|
| **adapt-to-anchor** (v1) | txt_h → frozen binary anchor | ~71 | inert (tie ≤0.2pt, W range 0.08–0.20) | BN+L2 + clean anchor codes |
| **code-defining** (v2) | img_h+txt_h from scratch | ~78 | inert (tie ≤0.3pt < seed noise, W range 0.24–0.56) | BN+L2 fix geometry |

The absolute jump (71→78) is the **regime** (code-defining learns better codes than adapting to a frozen anchor),
**not** the composition. In BOTH regimes loss composition and weight-tuning are inert: **BatchNorm dominates the
1-bit code geometry**. The AAAI claim "composition dominates tuning" is not supported for this BatchNorm'd
nested-hash head in either regime.

## Per gate-first rule
**Image and distillation fronts NOT started** — they reuse the same BatchNorm'd head and would be inert for the same
reason. A genuine test of "composition dominates" would require a head **without** the per-bit BatchNorm (so the
auxiliary terms can shape pre-sign geometry) — a different architecture, not a tuning change. Reported, no forcing.

## Repro
`web/loss_composition_v2.py --sweep --epochs 8 --out paper/loss_composition_v2.csv` on DGX (`.venv/bin/python`, GPU;
~34 min, incremental durable CSV, C+U emitted before W so a partial run still answers the gate). Frozen inputs:
`/tmp/emb_aug.pt` (train), `/tmp/emb_cache.pt` (test), `/tmp/hp_results.json` recipe, `/tmp/ft_ko_113.pt` arch config.
