# Does effective-dimension / entropy govern binary retrieval? — Part 1 (predictor test)

**Verdict: RED** (stopped before Part 2, per the gate rule). Effective-dimension / effective-rank do
**not** strongly predict binary R@10 across languages. Eval-only. Branch `paper-rank-governs`.
Data `paper/rank_predictor.csv`, figure `paper/fig_rank_predictors.pdf`. Run `scripts/rank_predictor.py`.

## Setup
Anchor ft113; ceiling/distill/head-adapt (COCO 5K) + 36 languages (XM3600, ceiling path).
Per row: **upstream_effdim** (text-emb PCA participation ratio — computed without codes, actionable),
**code_effrank** (participation ratio of the 1024-bit code matrix), **code_entropy** (Σ per-bit entropy),
and proxies **cosine** (pair text-image), **margin** (mean|z|), **diralign** (code↔paired-image-code, = pair-Hamming-derived).

## Cross-language Spearman ρ vs R@10 (n=36) — the gate test
| variable | type | ρ |
|---|---|---|
| **upstream_effdim** | upstream / actionable | **+0.376** |
| **code_effrank** | downstream (effective rank) | **+0.121** |
| code_entropy | downstream (tautology-risk) | +0.729 |
| cosine | proxy (embedding) | +0.788 |
| margin (mean\|z\|) | proxy | −0.866 *(spurious — see below)* |
| diralign | proxy (pair-Hamming-derived) | +0.980 *(tautological)* |
| code_rank (hard) | — | n/a (always 1024) |

**GREEN required**: upstream effdim *or* code rank with ρ≳0.8, and the 5 proxies weak. **Neither holds.**

## Why RED (honest reading)
- **Upstream effective-dim is a weak cross-language predictor (ρ=0.38)**, not the governing variable. Anatomy #11's "low effdim → weak language" is real only at the extreme tail (mi 37.3→R 1.7, te 51→6, quz 49→9); across the full set it does not order R@10 — counterexamples: sv effdim 55→R 85, fil 82→R 40, el 80→R 71, ko 89→R 84. So **#11 should be downgraded** from "governing variable" to "extreme-tail co-occurrence."
- **Code effective-rank is ~uncorrelated (ρ=0.12)**; hard rank is constant (1024) → carries no signal.
- **The proxies are not uniformly weak.** `cosine` (paired text-image cosine — a *previously-"failed"* proxy in the cross-path paradox) is the strongest **non-tautological, non-downstream** predictor cross-language (ρ=0.79). `diralign` (ρ=0.98) is tautological (it is 1−pair-Hamming/512). `margin` (ρ=−0.866) is **spurious**: mean|z| spans only 0.02492–0.02501 (BN+L2 pins it), so the correlation is on 4th–5th-decimal noise.
- **code_entropy (ρ=0.73)** is moderate but downstream/tautology-adjacent and tail-driven (mi codeH 668, te 808 vs strong ~990); not the actionable upstream variable the hypothesis sought.

## Cross-path note (n=3, anecdotal)
Across the 3 paths effdim *does* order with R@10 (ceiling 76.7/R 79.9 > head-adapt 58.5/R 74.0 > distillation 41.9/R 70.9). But n=3 is anecdotal and does **not** survive the 36-language test — the robust gate fails.

## en/ko outlier check (requested)
en effdim 61→R 82, ko effdim 89→R 84 are not outliers in the low-effdim/high-R sense; the real spread is that *many* languages have mid-high effdim with widely varying R@10, i.e. effdim simply isn't the controlling axis. The cleanest single cross-language predictor here is **semantic alignment (cosine)**, not representational rank.

## Decision
Part 1 gate failed (upstream effdim ρ=0.38, code effrank ρ=0.12, both ≪ 0.8). Per the brief
("Part1 red → 멈추고 보고"), **Part 2 (rank-preservation head retrain) is NOT entered.**
This is a clean negative: *representational effective-dimension is not the hidden governing variable*;
across languages, retrieval tracks semantic embedding alignment (cosine) and the tautological code-space
separation, while rank/effdim are at best tail-symptoms. (10th gate; first to test a positive-signal
hypothesis — refuted as a strong predictor.)
