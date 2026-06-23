# Anatomy — Backbone → Head → Code (neutral observations)

Exploration, not gating. Frozen ft113 anchor (`/tmp/ft_ko_113.pt`), three text paths into the
shared image-anchor code space: **ceiling** (true SigLIP-text→txt_h), **distillation**
(e5→distill→pred SigLIP-text→txt_h), **head-adapt** (e5→txt_h'). pre-sign `z = L2norm(BN(hash_head(x)[:,:1024]))`,
code = sign(z). COCO 5K test (A–D) + XM3600 36-lang (E). All figures have a backing CSV.
Descriptions below are *what the distribution looks like* — interpretation deferred.

Reproduce: `.venv/bin/python scripts/anatomy_extract.py && scripts/anatomy_plot.py` (A/B/C);
`scripts/anatomy_de.py && scripts/anatomy_plot_de.py` (D/E). Data: `paper/anatomy_*.csv`, figs `paper/fig_anatomy_*.pdf`.

## A — backbone embedding space (pre-head)
- **A1 svspectrum / A2 perdim_var**: effective dim (participation ratio): image **99.8**/1152, SigLIP-text (ceiling) **76.7**, distillation-output **41.9**, e5 (head-adapt) **58.5**/384.
- **A3 norm_hist**: raw L2 norms — image 11.1–19.8 (med 16.3), SigLIP-text 20.1–46.9 (med 27.9); distillation & e5 are pre-normalized (=1.0).
- **A4 cosine**: paired vs random cross-modal cosine (SigLIP space) for ceiling & distillation.
- **A5 scatter (PCA+UMAP)**: image points form a single cluster fully disjoint from text; ceiling & distillation text overlap each other.

## B — head internal stages (sliced → BN → L2 z)
- **B1 stage_var / B2 stage_hist**: per-dim variance and value distribution at each stage, all 4 paths.
- **B3 absz**: mean|z| = **0.0249 for all four paths** (ceiling/distill/head-adapt/image identical to 4 d.p.); frac|z|<0.01 ≈ 0.25 for all.

## C — code space (post-sign, 1024b)
- **C1 bitbalance**: per-bit +1 fraction tightly around 0.5; summed per-bit entropy 1018–1023 / 1024 (all paths).
- **C2 bitcorr**: bit-bit correlation (first 256b heatmap); mean |off-diagonal corr| 0.075 (image) – 0.082 (head-adapt); all code matrices full rank (1024).
- **C3 hamming**: median paired vs nearest-wrong Hamming — ceiling 226 / 219 (gap −11), distillation 234 / 210 (gap −26), head-adapt 236 / 216 (gap −20).
- **C4 ablation**: R@10 drop when each contiguous 64-bit group is removed (per path). full R@10 = ceiling 79.92, head-adapt 74.00, distillation 70.86.

## D — path comparison
- **D1 bitflip** (sign flips vs ceiling true-code): distillation flips **13.1%** of bits (per-bit 7.9–16.9%), head-adapt **18.4%** (10.1–26.6%). "Confident" flips (flip AND |z|>median): distillation 1.0%, head-adapt 2.9%.
- **D2 success_features** (success vs failure query means): caption length nearly equal (succ ~10.5 vs fail ~10.2 words, all paths); the separating feature is paired Hamming / |z| (see CSV). `anatomy_D_fail_overlap.csv` gives per-path and all-path failure counts.

## E — multilingual / on-device (XM3600, ceiling path, 36 langs)
- **E1 lang_hamming_vs_r10**: median paired text-image Hamming vs R@10 is near-monotonic across all 36 languages; **Latin and non-Latin are intermixed along the same curve.** Strong: de 227b/92.6, fr/it 228/91.5, ru 231/91.3, id 231/91.0, vi 234/90.2. Weak tail: th 296/57.9, hi 314/43.4, fil(latin) 330/40.1, bn 354/34.8, sw(latin) 394/22.0, te(latin) 465/6.2, quz(latin) 473/8.7, mi 505/1.7.
- **E2 lang_umap**: per-language text-embedding UMAP (10 sampled langs).
- **E3 int8_flip**: int8 quantization sign-flip fraction per bit — mean **0.0090**, max 0.0148.

## Notable / unexpected (observations, flagged for joint interpretation — NOT conclusions)
1. **mean|z| is identical (0.0249) across all four paths** — the BN+L2 stage pins the pre-sign scale to the same value regardless of input source (A/B).
2. **Distillation output has the lowest effective dimension (41.9)** — markedly more anisotropic than true SigLIP-text (76.7) and image (99.8), despite living in the same 1152-D space (A1).
3. **Code-space health is path-independent**: all three text paths produce near-balanced, near-max-entropy, full-rank, low-correlation 1024-bit codes (C) — the large R@10 gaps (79.9 / 74.0 / 70.9) are not visible in these aggregate code-quality stats.
4. **head-adapt flips MORE bits from the ceiling code (18.4%) than distillation (13.1%), yet retrieves better (74.0 vs 70.9)** — agreement with the ceiling code does not order R@10 (D1 vs C4).
5. **Median Hamming separation gap (nearest-wrong − pair) orders with R@10**: ceiling −11 > head-adapt −20 > distillation −26 (C3) — the one aggregate that matches the ranking.
6. **Multilingual strength is set by code-distance-to-anchor, not script**: Latin sw/te/quz fall in the weak tail while Cyrillic ru is among the strongest; the curve is continuous in pair-Hamming (E1).
7. **int8 quantization is nearly lossless at the bit level** (≤1.5% flip on any bit; ~0.9% mean) (E3).

## D-deep (follow-up figures D5/D6)
- **D5 flip_vs_absz**: flip-fraction vs ceiling, binned by pre-sign |z| decile. For BOTH paths it **decays monotonically with |z|** — distillation 0.437 (smallest-|z| decile) → 0.001 (largest); head-adapt 0.459 → 0.008. head-adapt flips more than distillation at *every* decile. CSV `anatomy_Ddeep_flip_vs_absz.csv`.
- **D6 gap_success_fail**: per-query separation gap (nearest-wrong − pair) split by R@10 success/failure. Success median ≈ +2 (ceiling) / −7 (distill) / −6 (head-adapt); **failure median ≈ −81 / −78 / −73** (the paired image is ~75–87 bits *farther* than the nearest wrong one). Failure counts: ceiling 1004, distillation 1457, head-adapt 1300. CSV `anatomy_Ddeep_gap.csv`.

### Notable (D-deep)
8. **Sign flips are boundary-concentrated, not "confident"** — for both distillation and head-adapt the flip rate collapses toward 0 as |z| grows (≤1% at the top |z| decile). This *refines/contradicts* the earlier "distillation confidently maps to a different sign" framing: the divergence from the ceiling code lives at small margins, not at high-confidence bits. head-adapt simply has a uniformly higher boundary-flip rate.
9. **The success/failure boundary is the separation gap, and it is bimodal-by-outcome** — within each path the gap is near-0 for hits and ≈ −80 for misses (a clean split), and head-adapt wins largely by having *fewer* such large-negative-gap queries than distillation (1300 vs 1457), not by a different success-mode gap.

## F-cuts (deeper neutral, figs F1/F2)
- **F1 cumvar / residual** (distillation collapse): dims for 90% variance — image (high) / ceiling **241** / distillation **76** (99%: ceiling 546 vs distillation 141). Cross-subspace energy: distillation has **76.6%** of its variance inside ceiling's top-50 PCs (ceiling itself only 54.9% in distillation's top-50); image has just **25.5%** energy in ceiling's top-50 (modality gap). CSVs `anatomy_F1_cumvar.csv`, `anatomy_F1_residual.csv`.
- **F2 lang_geometry** (XM3600 per-language): code bit-entropy, text-emb effective-dim, centroid-Hamming-to-image, pair-Hamming spread, all vs R@10. CSV `anatomy_F2_lang_geometry.csv`.

### Notable (F)
10. **Distillation lives in a lower-dim subspace that is *inside* the ceiling text subspace** (90%-var in 76 dims vs 241; 77% energy in ceiling's top-50). It is not off-manifold noise — it is a compressed/contracted version of the same dominant text directions.
11. **Weak-language text codes are degenerate (low bit-entropy), and this tracks R@10 monotonically** — strong langs use ~990–1004/1024 bits of entropy; the weak tail collapses: mi 668, te 808, quz 853, sw 940, hi 958, bn 953. It originates upstream in the embedding: the same weak langs have low text-embedding effective-dim (mi 37.3, te 51.1, quz 49.1 vs de 93, fr 84). The per-language code centroid distance to the image centroid is nearly constant (~496–500 soft-Hamming) and only slightly elevated for the very weakest (mi 513) — i.e. the weak-language signal is *low-entropy/contracted codes*, not a uniformly shifted centroid.
