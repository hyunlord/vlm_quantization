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

## RESULTS
_pending sweep completion._

## VERDICT
_pending._
