# Sign-loss quantification + outlier debug (decision data)

Branch `analysis-signloss-outlier`. Eval-only, COCO 5K T2I, gold=diagonal instance, paths ceiling/distill/head-adapt.
Data: `paper/signloss_quant.csv`, `paper/outlier_debug.csv`, `viz/data/{signloss,outlier_cases}.json`,
viz `viz/signloss.html`, `viz/outlier_gallery.html` (+ `viz/data/thumbs_outlier/`).

## Part 1 — sign information loss → A-bet GO/NO-GO
Three retrieval modes (gallery=image): **continuous** z_q·z_g (ceiling before sign) · **asymmetric** z_q·b_g
(query continuous, **gallery stays 1-bit = deployable**) · **hamming** b_q·b_g (current). EN T2I R@10, ceiling path:

| bit | continuous | asymmetric | hamming | gap cont−ham (sign loss) | gap asym−ham (search-only) |
|---|---|---|---|---|---|
| 64 | 80.42 | 74.70 | 66.52 | **+13.90** | +8.18 |
| 128 | 80.36 | 77.94 | 74.10 | +6.26 | +3.84 |
| 256 | 80.48 | 79.38 | 77.60 | +2.88 | +1.78 |
| 512 | 80.54 | 80.24 | 79.18 | +1.36 | +1.06 |
| 1024 | 80.52 | 80.54 | 79.92 | **+0.60** | +0.62 |

(distillation & head-adapt show the identical shape: gap c−h ≈ 15→1pt as bits 64→1024.) top-10 overlap
(cont vs ham) rises 0.52→0.85; Kendall τ 0.45→0.77 with bits.

**Two facts:**
1. **The continuous ceiling is ~bit-flat (~80.4 even at 64-d).** The retrieval-relevant information lives in
   a handful of continuous dimensions; **binarization, not dimensionality, is the low-bit bottleneck.**
2. **Sign loss is strongly bit-dependent**: ~14pt at 64-bit, ~0.6pt at 1024-bit.

### Verdict — bit-dependent (not a single GO/NO-GO)
- **At 1024-bit (current headline deployment): NO-GO.** Sign costs only +0.6pt; the sign code is already
  near-sufficient (high-dim balanced sign ≈ cosine). A code-alphabet change won't move the 1024-bit number.
- **At low/mid bits (64–256): GO.** Sign discards +3 to +14pt — exactly the **small-code / on-device regime**
  where storage matters most. A multi-bit / residual-PQ / learned-codebook alphabet has real room here.
- **Asymmetric search = free partial-GO** (no retraining, **gallery unchanged at 1-bit**): recovers ~half the
  low-bit gap (+8.2pt@64, +3.8@128, +1.8@256), negligible at 1024 (+0.6). A deployable, search-only lever for
  short codes. *Honesty: the full continuous ceiling needs continuous storage (not 1-bit) → not deployable as-is;
  only the asymmetric variant is.*

**Action implied:** if the bet is "better 1024-bit accuracy" → NO-GO (look to backbone). If "small on-device
codes" → GO for a sign-replacement alphabet **and** asymmetric search at 64–256 bit.

## Part 2 — q19-type outliers → metric correction vs new failure mode
Among queries whose gold is **not in top-10** (failures), split by whether the rank-1 retrieved (wrong) image
is **close** (Hamming < 226, global median paired) or **far**:

| path | fail % | confusable % (close wrong = label-limit cand A) | isolated % (far wrong = q19 cand B) | fail gold-ham med | fail top1-ham med |
|---|---|---|---|---|---|
| ceiling | 20.1 | **12.8 (≈64% of fails)** | 7.3 (≈36%) | 297 | 212 |
| distillation | 29.1 | 20.3 | 8.8 | 289 | 206 |
| head-adapt | 26.0 | 16.5 | 9.5 | 296 | 213 |

**Reading (candidate, needs human confirmation via thumbnails):**
- The **majority of failures are "confusable"** — a *closer* image (Hamming ~150–210 vs gold ~297) is retrieved
  first. If those close images are **semantically correct** (e.g. another snowboard photo for "a man riding a
  snowboard"), then **instance R@K is penalizing semantic near-duplicates** → there is room for a **relaxed /
  category-aware recall** reported alongside instance R@K (a metric-correction contribution, candidate **A**).
- The **isolated "q19" type is real but the minority (~7% of all queries, ~36% of fails)** — even the rank-1 is
  far (no good match); these are genuinely poorly-placed query codes (candidate **B**, a smaller new-failure bucket).
- **Decision needs eyeballing**: `viz/outlier_gallery.html` shows, per case, the query caption + gold thumb +
  top-3 retrieved thumbs. If the confusable wrongs look semantically right → pursue relaxed-recall metric. CC
  flags candidates only; the A/B call is the human's.

## Repro
`scripts/signloss_quant.py` → Part 1 CSV/JSON; `scripts/outlier_debug.py` → Part 2 CSV/JSON + thumbnails
(needs `data/coco/dataset_coco.json` + `data/coco/{val2014,train2014}/` on DGX). Viz: `cd viz && python -m http.server`.
