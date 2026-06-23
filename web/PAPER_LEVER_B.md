# Lever B — Head Architecture Redesign / BN-free (`paper-lever-arch`)

**Status: RESULTS PENDING** (25-epoch × 5-bit × 2-seed sweep running on DGX).
This file records setup, grounding, and gate definition; verdict/numbers filled on completion.

## Premise
Across both loss-composition regimes the auxiliary terms were **inert** (≤0.2pt at 1024b,
inside seed noise), and every doc attributed this to the **per-bit BatchNorm + L2** fixing the
pre-sign code geometry (`paper-loss-composition` f0af984, `-v2` 0ea0dff). The v2 doc states
explicitly: *"A genuine test of 'composition dominates' would require a head without the per-bit
BatchNorm — a different architecture, not a tuning change."* No BN-free head was ever trained.
Lever B builds exactly that head and asks whether the bottleneck moves.

## What is tested (untested before)
Head = `NestedHashLayer` trunk `Linear→LayerNorm→GELU→Dropout→Linear`, then per-bit normalization
swapped via `HEAD_NORM`:
- **bn** — current head (baseline): `F.normalize(BatchNorm(sliced))`.
- **ln** — `F.normalize(LayerNorm(sliced))` (norm without batch statistics).
- **none** — `F.normalize(sliced)` (pure L2, no per-bit norm).
- **none_scale** — learnable per-dim affine then L2 (isolates BN's affine from its batch-normalization).
- **rotation** — learnable rotation on raw (ITQ-style) + soft orthogonality penalty, then L2 (no BN).
- **none + infonce_hmargin** (B-d) — BN-free × Lever-A loss interaction.

Plus the **composition-response arm**: `AUX_SCALE ∈ {0,1,4}` on both `bn` and `none` — does the
aux-loss weight finally move R@10 once BN is removed?

All head-only on cached SigLIP2 embeddings, mode = coco, baseline = `bn-infonce`.

## Gate
- **GREEN** iff (i) some variant gives **R@10 ≥ +1.0pt over baseline** (≥3 bits, beyond noise),
  **OR** (ii) under BN-free the loss composition **finally responds ≥0.5pt** to aux weight/config
  (vs the prior inert ≤0.2pt) — the latter is a new "normalization–loss interaction" analysis contribution.
- **RED** iff BN-free underperforms (BN is essential) AND composition stays inert → **strengthens**
  the "BN dominates the code geometry" conclusion.

## Method / reproduction
- Harness `scripts/lever_sweep.py` (configurable head); driver `scripts/lever_run.sh`; report `scripts/lever_report.py`.
- Same eval as Lever A (COCO 5K, EN T2I+I2T + KO T2I, R@{1,5,10}, bit-sweep).
- Stability watched (`naninf` counter); lr/warmup = baseline OneCycle (pct_start 0.3).

## Deployability note (one line, pre-committed)
All head-norm variants keep the **encoder unchanged** and the head the same tiny MLP → **zero**
deployment cost (BN/LN/affine/rotation all fold into inference). Fully ort-web / on-device compatible.
`rotation` mildly breaks strict prefix-nesting (full-dim mix before slice) — noted caveat for Matryoshka use.

## Early (non-verdict) signal
3-epoch poke: `none`, `none_scale`, `rotation` all train **stably** (naninf 0) and are competitive at
1024b (R@10 77.8–78.2 at 3 epochs) — removing per-bit BN does **not** break training. Whether it
*helps* (and whether composition then responds) awaits the full sweep.

## RESULTS
_pending sweep completion._

## VERDICT
_pending._
