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
