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

## RESULTS (2-seed mean, EN T2I R@10; head-only, mode=coco, EPOCHS=15)
Baseline `bn-infonce@15`: 64=68.01 128=75.10 256=78.10 512=79.66 1024=80.73. All variants train **stably (naninf 0)** — no lr/warmup changes needed; BN-free does NOT diverge.

| head-norm | 64 | 128 | 256 | 512 | 1024 | ΔR@10 vs bn (per bit) |
|---|---|---|---|---|---|---|
| ln | 68.42 | 75.06 | 78.30 | 79.68 | 80.20 | +0.41 / −0.04 / +0.20 / +0.02 / −0.53 |
| none (BN-free) | 68.10 | 74.76 | 78.08 | 79.64 | 80.24 | +0.09 / −0.34 / −0.02 / −0.02 / −0.49 |
| none_scale (learn affine) | 67.88 | 74.92 | 77.84 | 79.42 | 80.26 | −0.13 / −0.18 / −0.26 / −0.24 / −0.47 |
| rotation (ITQ-style) | 67.34 | 74.44 | 77.38 | 79.40 | 80.20 | −0.67 / −0.66 / −0.72 / −0.26 / −0.53 |
| none + hmargin (w10, B-d) | 68.02 | 74.56 | 77.98 | 79.62 | 80.46 | +0.01 / −0.54 / −0.12 / −0.04 / −0.27 |

Every variant is within ±0.7pt of baseline at every bit (max +0.49 at one bit/seed); LN≈none≈affine≈baseline, rotation slightly **worst**, all ~0.5pt below baseline at 1024b.

**Composition-response arm (gate-2): does aux weight move R@10 once BN is removed?** (EN T2I R@10 @1024)
- `bn`: aux0=80.86, aux1=80.73, aux4=80.25 → spread 0.61pt, but **aux HURTS** (more aux = lower R@10) — not "the loss finally works".
- `none` (BN-free): aux0=80.66, aux1=80.54, aux4=80.00 → spread 0.66pt — also **aux HURTS, slightly MORE** than under BN (aux4: none 80.00 < bn 80.25).

So removing BN does **not** unlock beneficial composition — the aux terms (ortho/quant/balance/cons/lcs) are at best neutral, at worst harmful at 1024b, *with or without* BN, and BN if anything **buffers** the harm.

## VERDICT — **RED** (both gate arms)
- Gate-1 (R@10 ≥ +1.0pt): no variant; all ≈ baseline within noise, rotation slightly worse. RED.
- Gate-2 (does BN-free unlock *beneficial* composition?): no. Both bn and none "respond" ≥0.5pt to aux weight but in the **negative** direction (aux hurts), none more than bn. The "BN suppresses a useful loss" hypothesis is refuted. RED.

**Message (strengthens analysis):** per-bit BatchNorm is **one valid normalizer among several** — LayerNorm, pure-L2, and learnable-affine all land within ~0.5pt, and BN-free trains stably (so BN is **not essential**, correcting any "BN is the load-bearing trick" assumption). But removing it confers **no benefit and does not unlock composition** → the inertness of auxiliary losses is **not caused by BN**; the code geometry is set by **InfoNCE alignment + L2 normalization**, which any reasonable per-bit normalizer leaves intact. Learnable rotation (ITQ-style) does not help and slightly breaks Matryoshka nesting. Deployability: all variants encoder-unchanged, zero deploy cost — but none worth shipping over BN.
Final: 2 seeds complete (165-row sweep, composition arm bn+none × aux{0,1,4}); RED confirmed.
