# PAPER_PILLAR2_HAMMING — gate v2: Hamming-space predictors · RED (method-hunt ends)

Branch `paper-pillar2-hamming`. After the quadruple negative (cosine/parity/margin fail to predict R@K), the last
method-hunt: does a quantity computed **directly from the binary codes** (post-sign, Hamming-space) predict R@10
across paths/bits AND explain the paradox head-adapt (R 74.0) > distillation (R 70.86)? Driver `web/pillar2_hamming.py`,
eval-only on §3 paths (ceiling/distillation/head-adapt), gallery = ceiling image codes. **Verdict: RED** — no
*non-tautological* code-space proxy explains retrieval; the only predictors are restatements of the retrieval outcome.

## Results — `paper/pillar2_hamming_predictors.csv` (9 = 3 paths × 3 bits)

| predictor (kind) | Spearman ρ vs R@10 | head-adapt vs distillation @1024 | verdict |
|---|---|---|---|
| (a) jaccard@10 / @100 — neighbor preservation vs ceiling | 0.64 / 0.70 | 0.483 < 0.548 → **WRONG** | structural, **fails paradox** |
| (d) diralign — bit-match to *paired image* − avg | −0.48 | 0.2632 < 0.2643 → **WRONG** | structural, **fails paradox** |
| parity (prior, §3) | 0.14 | 81.6 < 86.9 → WRONG | fails (known) |
| cosine (prior, §3) | — | 0.866 (distill) only | fails (known) |
| margin D (prior, pillar2-margin) | 0.25 | — | fails (known) |
| (c) hmargin_mean — d(nearest-wrong) − d(true) | −0.65 (bit-confounded) | −23.2 > −28.0 → CORRECT | **tautological** |
| (c') hmargin_frac_pos — P(true closer than ALL wrong) | **1.00** | — | **tautological (≈ R@1)** |

Per-query @1024 (`paper/pillar2_hamming_perquery.csv`): pointbiserial(success, hmargin) = **0.69**, (success, diralign)
= 0.64. (hmargin's per-query power is definitional — hmargin>0 *is* rank-1 success.)

## Gate verdict: RED
1. **The genuinely-prior structural proxies fail the decisive paradox test.** Neighbor-Jaccard (ρ=0.70 across points)
   and direction-alignment-to-target both rank **distillation above head-adapt** — the *wrong* direction, same failure
   mode as parity. They do not explain why head-adapt retrieves better.
2. **The only quantities that explain the paradox / predict per-query are tautological.** hmargin (= d(nearest-wrong)
   − d(true)) and hmargin_frac_pos (≈ R@1) use the true-vs-wrong gallery distances — i.e. they *are* the retrieval
   measurement, not a cheap prior proxy (the brief excludes answer-rank-class quantities for exactly this reason).
3. **Decisive mechanistic finding:** absolute alignment to the target image code (diralign) is *slightly higher* for
   distillation (0.2643 > 0.2632) yet its R@10 is **lower** — so retrieval is not governed by closeness to the target
   but by **separation from negatives** (hmargin: head-adapt −23.2 > distillation −28.0). Head-adapt wins because
   InfoNCE pushes the code away from *wrong* gallery items, not because it lands closer to the right one. That
   separation is computable only against the gallery's negatives = the retrieval objective itself.

## Conclusion — method-hunt over; analysis/system paper confirmed
The quintuple negative now stands: **cosine ✗ · parity ✗ · margin ✗ · neighbor-Jaccard ✗ · direction-alignment ✗**
all fail to predict (or invert on) learned-binary-code retrieval. The only thing that tracks R@K is contrastive
separation from gallery negatives — which is exactly what end-to-end InfoNCE optimizes and is **not a pre-computable
proxy**. Optimizing it directly = InfoNCE (already deployed). **No novel proxy-loss method is warranted** (Phase 3 not
entered). The strong analysis-paper claim: *the retrieval quality of a learned binary code is not predicted by any
embedding-, confidence-, or code-structure proxy; it is achievable only by direct contrastive alignment against the
retrieval gallery.* (Protects the schedule: no fruitless proxy-loss build; positions the paper as analysis+system,
WACV/AAAI, not a CVPR method.)

## Repro
`web/pillar2_hamming.py` on DGX (`.venv/bin/python`, GPU; ft_ko_113 + emb_cache + distill_e5 + txt_h_e5 + e5_test_en;
Hamming via GPU matmul on ±1 codes H=(B−Q·Gᵀ)/2; ~3 min). Figures: `web/plot_pillar2_hamming.py`. Honest negative,
no forcing — every structural proxy reported alongside cosine/parity/margin.
