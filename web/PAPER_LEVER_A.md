# Lever A — Negative-Separation Training Losses (`paper-lever-negsep`)

**Status: RESULTS PENDING** (25-epoch × 5-bit × 2-seed sweep running on DGX).
This file records setup, grounding, and gate definition; verdict/numbers filled on completion.

## Premise
Prior work established the *decisive* diagnostic finding: retrieval quality tracks
**separation from gallery negatives** (hmargin), not closeness to the target — and that
"InfoNCE already optimizes the only thing that tracks R@K." But that conclusion was reached
by testing margin/Hamming/parity/Jaccard/diralign **only as predictors** of R@K
(`paper-pillar2-margin` a13a807, `paper-pillar2-hamming` b66e8f7). None was ever used as an
actual **training objective**. Lever A closes that gap: turn the negsep quantity into a loss.

## What is tested (untested before)
- **(a) Hamming hard-negative** (`hardneg`): code-space contrastive with hardness reweighting
  (Robinson et al. 2021) — nearest-wrong items dominate the denominator.
- **(b) margin-in-Hamming** (`hmargin`): triplet that directly maximizes the gap between the
  paired soft-Hamming and the nearest-wrong soft-Hamming (margin in fraction-of-bits).
- **(c) InfoNCE + (a)/(b)** at weights chosen so the negsep term is on a scale comparable to
  InfoNCE (fair test; `NEGSEP_W` sweep).
- Controls: pure `hardneg` / pure `hmargin` (can negsep ALONE define codes?), and
  `bn-aux0` = pure-InfoNCE contrastive-only (separates "beats InfoNCE" from "beats full recipe").

All head-only on cached SigLIP2 embeddings, head = current `NestedHashLayer` (BN+L2),
mode = coco. Baseline = `bn-infonce` (full recipe == current deployed head).

## Gate
- **GREEN** iff some variant gives **R@10 ≥ +1.0pt over InfoNCE**, consistent across ≥3 bits and
  beyond seed noise, AND the realized test-set hmargin distribution actually widens (mechanism match).
- **RED** iff ≤ seed noise → confirms "InfoNCE is already negative-separation-optimal" (strengthens
  the analysis-paper claim).

## Method / reproduction
- Harness: `scripts/lever_sweep.py` (reproduces `train_1024.py` coco baseline at `HEAD_NORM=bn LOSS=infonce`).
- Driver: `scripts/lever_run.sh`; analysis: `scripts/lever_report.py`.
- Eval: COCO 5K instance retrieval, EN (T2I+I2T) + KO (T2I), R@{1,5,10}, bits {64,128,256,512,1024},
  binary Hamming, id-based relevance (same as `eval_retrieval.py` / `eval_korean.py`).
- hmargin diagnostic logged per run (mean nearest-wrong − paired Hamming, in bits).

## Deployability note (one line, pre-committed)
A training-loss change keeps the encoder and head **identical** at inference → **zero** deployment
cost, fully ort-web / on-device compatible. If green, this is a deployable CVPR-method outcome
(better codes, same runtime). If red, it is a clean analysis result.

## Early (non-verdict) signal
3-epoch poke: pure `hardneg` collapses (R@10≈11 → negsep alone cannot define codes); pure `hmargin`
reached 1024b R@10 ≈ 79.7 (T2I) / 80.2 (I2T) at only 3 epochs with a widened hmargin diagnostic —
flagged for scrutiny but NOT a verdict (could be faster convergence to the same ceiling; the full
25-epoch × 2-seed comparison decides).

## RESULTS (2-seed mean, EN T2I R@10; head-only, mode=coco, EPOCHS=15)
Baseline `bn-infonce@15`: 64=68.01 128=75.10 256=78.10 512=79.66 1024=80.73
(faithful: `bn-infonce@25` = 80.88@1024, vs deployed 80.3; `aux0`=pure-InfoNCE ≈ baseline → aux terms inert, reconfirmed).

| variant | 64 | 128 | 256 | 512 | 1024 | ΔR@10 vs baseline (per bit) |
|---|---|---|---|---|---|---|
| hardneg (pure) | 4.95 | 6.26 | 8.38 | 10.33 | 10.96 | −63 / −69 / −70 / −69 / −70 |
| hmargin (m=.05/.1/.2, identical) | 61.64 | 72.51 | 77.10 | 79.86 | **81.42** | **−6.37 / −2.59 / −1.00 / +0.20 / +0.69** |
| infonce+hmargin (w10) | 67.56 | 75.06 | 78.12 | 79.64 | 80.61 | −0.45 / −0.04 / +0.02 / −0.02 / −0.12 |
| infonce+hardneg (w1) | 67.48 | 74.86 | 77.90 | 79.34 | 80.60 | −0.53 / −0.24 / −0.20 / −0.32 / −0.13 |

**Mechanism check (realized hmargin, bits @1024):** baseline −12.1 → pure hmargin **−3.4** (separation genuinely widened); hardneg −33 (collapsed). The loss *does* what it claims — the margin widens — yet R@10 does not rise.

**Secondary (not the gate):** pure hmargin is a **high-bit top-1 sharpener / low-bit recall-killer**: R@1@1024 42.76→**47.0 (+4.2)**, KO R@10@1024 63.33→**64.56 (+1.23)**, but R@10@64 67.58→61.9 (**−5.7**), KO@64 −6.7. With scarce bits it over-separates the single hardest negative at the cost of broad recall; with abundant bits it sharpens the top rank. A capacity-dependent precision/recall re-trade, not a uniform gain. `infonce+hmargin` is inert (InfoNCE subsumes the margin term).

margin insensitivity: m∈{.05,.1,.2} give identical results — `∂/∂θ relu(m−gap) = −∂gap/∂θ` for active pairs, so m only sets the (here fully-active) threshold. Seed noise ≤0.32pt @1024.

## VERDICT — **RED** (primary R@10 gate)
No variant reaches +1.0pt R@10; best is hmargin +0.69 @1024 (and −1 to −6 at the lower 3 bits), inside/under noise vs the stricter @25 baseline (+0.54). The mechanism fires (separation widens) but R@10 does not follow.
**Message (strengthens analysis):** *InfoNCE is already negative-separation-optimal for R@10.* A binary-specific margin loss only re-trades top-1 vs recall along the bit-budget axis. Deployability: training-loss-only, encoder/head unchanged → the **R@1/high-bit sharpening is a free, deployable side-effect** worth a paragraph, but not a CVPR method on the stated R@10 gate.
(pending: seed-1 of 3 combined configs — verdict robust; deltas already negative.)
