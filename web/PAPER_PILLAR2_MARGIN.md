# PAPER_PILLAR2_MARGIN — margin theory gate (diagnosis → math) · NEGATIVE

Branch `paper-pillar2-margin`. Tests whether the pre-sign margin |z| distribution yields a **closed form** that
predicts binary retrieval (where §3 showed cosine *and* parity fail). Driver `web/pillar2_margin.py`, eval-only on
§3's three paths (ceiling / distillation / head-adapt) + §2's int8 noise. **Gate result: the closed form does NOT
exist — margin does not predict retrieval either. Do not enter Phase 2 with a margin-aware loss.** Honest negative.

## Setup
Three text paths into the frozen ft113 code space (gallery = img_h(test_img)), COCO 5K, I2T; pre-sign z =
L2norm(BN(hash_head(x))) (continuous, before sign). Noise σ reused from §2 (encoder-output int8): the int8-induced
z-perturbation std **σ = 0.00093** (tiny — z is L2-normed over 1024 dims, so per-coord |z| ≈ 0.025).

## (1) Margin distribution — `paper/pillar2_margin_dist.csv`
mean|z| ≈ **0.025 for all three paths** (1024-bit). At 64/256 bit head-adapt is *slightly thinner* (0.0238/0.0236)
than ceiling/distillation (0.0256/0.0255) — yet head-adapt retrieves **better**. **Hypothesis "head-adapt has thicker
margin" is NOT supported** (it's equal-or-thinner).

## (2) flip ≈ Φ(−|z|/σ) — `paper/pillar2_flip_vs_margin.csv`
Empirical sign-flip under int8 noise, binned by |z|, vs Gaussian Φ(−|z|/σ): **weighted R² = 0.81**. The Gaussian
captures the trend but is only approximate (per-bit noise is not iid Gaussian; σ varies by bit). Not a clean law.

## (3) ★GATE★ — does D = mean_k Φ(−|z_k|/σ) predict R@10? — `paper/pillar2_margin_predicts_R.csv`

| path | bit | D (exp. distortion) | mean|z| | cosine→true | parity% | bin R@10 |
|---|---|---|---|---|---|---|
| ceiling | 1024 | 0.0094 | 0.0250 | 1.000 | 100.0 | 79.92 |
| distillation | 1024 | 0.0095 | 0.0250 | 0.866 | 86.91 | 70.86 |
| head-adapt | 1024 | 0.0094 | 0.0250 | — | 81.64 | 74.00 |

(64/256-bit rows in CSV.) Across the 9 (path,bit) points:
**Spearman(D, R10) = 0.25** (weak, *wrong sign* — higher distortion should mean lower R), (mean|z|, R10) = **−0.30**,
(parity, R10) = 0.32. **None predicts retrieval.** D is ≈0.009 for *every* path (σ is so small that expected distortion
is tiny and nearly path-invariant), so it cannot explain the 9-pt R@10 spread (the spread is driven by bit-count
within a path and by alignment quality across paths, not by margin).

**Per-query** — `paper/pillar2_perquery.csv`: pointbiserial(retrieval success, per-query D_q) = **−0.013** (≈0).
Success rate by D_q quintile is flat: **82.4 / 78.3 / 79.9 / 79.9 / 79.1**. Per-query margin does **not** predict which
queries fail.

## (4) Causal mini-check
Under the realistic int8 noise, retrieval is **unchanged**: clean R@10 79.92 → orig+noise **79.92** → 5×-margin+noise
80.06. The deployment noise (σ=0.00093) is far too small to flip retrieval-relevant bits, so thickening the margin buys
nothing. (Consistent with §2: int8 is numerically harmless.)

## Gate verdict: RED (no closed form; margin is not the lever)
1. flip≈Φ(−|z|/σ) holds only approximately (R²=0.81).
2. The expected-distortion D **does not predict R@10** at code level (Spearman 0.25, wrong sign) or per-query
   (pointbiserial −0.01). mean|z| and parity also fail.
3. Under realistic noise, margin is causally inert (retrieval unchanged).

**Mechanism (the honest answer to "what governs retrieval"):** distillation's 13% parity gap is **systematic, not
stochastic** — the distilled embedding maps to a *confidently different* sign pattern (margins are normal, ~0.025),
not to low-margin bits that randomly flip. A noise model Φ(−|z|/σ) is therefore the **wrong model** for the gap, and
margin magnitude is uninformative. Retrieval is governed by code **direction** (which bits are set, relative to the
gallery in Hamming space) — exactly what head-adapt's InfoNCE-to-image-codes optimizes directly and what cosine/
parity/margin all fail to capture. This is why head-adapt wins with *lower* teacher-parity and *equal* margin.

## Implication
**Do NOT enter Phase 2 with a margin/parity-aware loss** — the data says it would not help (margin doesn't predict R).
The validated lever remains **direct code-space alignment** (head-adapt), already deployed. The Pillar-2 contribution
is the *triple negative*: cosine ✗, parity ✗, margin ✗ → retrieval quality of a learned binary code is not predicted
by any embedding- or confidence-space proxy, only by retrieval-space (Hamming-to-gallery) alignment. (Protects the
schedule: no fruitless margin-loss build.)

## Repro
`web/pillar2_margin.py` on DGX (`.venv/bin/python`, GPU; ft_ko_113 + emb_cache + distill_e5 + txt_h_e5 + e5_test_en;
loads fine-tuned e5; ~2 min). Figures: `web/plot_pillar2.py` → `paper/fig_pillar2_*.pdf`. seed: eval is deterministic;
the only stochasticity (int8 noise realization) is negligible (§4 causal row).
