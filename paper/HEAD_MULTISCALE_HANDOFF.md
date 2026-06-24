# Head-structure gate (multiscale / UNet-style) — handoff

**Question**: does a multiscale *information-exchange* / *depth-wise extraction* head beat the current
single-projection **prefix-slicing** `NestedHashLayer`? Each scale's code must stay **independently
searchable** (on-device prefix identity). Skeptical, no forced positives.

**Setup (fair internal harness)**: all variants share identical data / loss / scales / optimizer — only
the head structure differs, so **Δ-vs-baseline is the signal**. Frozen so400m embeddings (no backbone
change). Train = emb_cache[train] EN 8K + coco_ko_pairs KO 16K (L2-normed, `norm_in=1`); loss =
`CombinedHashLoss` (InfoNCE + EAQL + ortho + balance + LCS); scales {8,16,32,64,128,256,512,1024};
AdamW lr 1e-3, 30 epochs, batch 512; **2 seeds**. Eval = COCO-5K test, **per-scale independent**
Hamming T2I R@{1,10}, EN (emb_cache test) + KO (coco_ko_test, aligned). Code: `scripts/head_multiscale_train.py`,
`src/models/multiscale_heads.py`. Data: `paper/head_multiscale.csv` + `paper/head_multiscale_summary.json`.

> **Caveat (absolute level):** this lightweight harness (8K EN / 30 ep) lands the baseline at
> 1024-bit R@10 EN ≈ **64** / KO ≈ **55**, *below* the deployed ft113 (~80 / ~71) which uses 113K+ data.
> The gate is about **relative structure**, not absolute SoTA. A lower-data regime gives novel structure
> *more* headroom to help, so an inert/negative result here is a **strong** RED.

## Per-scale R@10 (T2I, mean of 2 seeds; baseline alongside)

Baseline `NestedHashLayer` (842K params, 1.5 ms/1k):

| scale | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| EN R@10 | 9.0 | 18.5 | 32.3 | 46.4 | 55.1 | 59.7 | 62.5 | **63.9** |
| KO R@10 | 7.5 | 16.7 | 27.7 | 39.3 | 46.8 | 51.4 | 53.9 | **55.4** |
| seed noise (EN, ±) | ~0.3 | ~0.5 | ~0.1 | ~0.4 | ~0.2 | ~0.2 | ~0.0 | ~0.1 |

Δ vs baseline at 1024-bit (EN R@10), mean of 2 seeds:

| Gate | Variant | params | 1024 EN | Δ1024 EN | 1024 KO | Δ1024 KO | verdict |
|---|---|---|---|---|---|---|---|
| — | baseline (prefix-slice) | 842K | 63.9 | — | 55.4 | — | reference |
| 1 | conv (1D-conv stack) | 2175K | 0.2 | **−63.7** | 0.2 | −55.2 | 🔴 RED (collapse) |
| 1 | conv-UNet | 3947K | 0.2 | **−63.7** | 0.2 | −55.2 | 🔴 RED (collapse) |
| 3 | depth (depth-only, no skip) | 1827K | 49.0 | **−14.9** | 42.9 | −12.5 | 🔴 RED |
| 3★ | depth+skip (UNet exchange) | 2602K | 55.0 | **−8.9** | 47.4 | −8.0 | 🔴 RED vs base; **>depth-only** |
| 3 | depth+topdown (coarse→fine) | 2526K | 49.0 | **−14.9** | 42.8 | −12.6 | 🔴 RED (≈depth-only) |
| 2 | residual (coarse-to-fine) | 2019K | 62.3 | **−1.6** | 54.5 | −0.8 | 🔴 RED (≈baseline, inert) |

Per-scale Δ EN R@10 (mean of 2 seeds), coarse→fine: baseline noise ±0.06–0.38.

| variant | Δ8 | Δ64 | Δ256 | Δ1024 |
|---|---|---|---|---|
| conv / conv-UNet | −8.7 | −46.2 | −59.4 | −63.7 |
| depth | −1.3 | −8.2 | −13.0 | −14.9 |
| depth+skip | −1.5 | −7.3 | −8.9 | −8.9 |
| depth+topdown | −1.7 | −8.2 | −12.1 | −14.9 |
| residual | −0.4 | −0.8 | −1.0 | −1.6 |

Every variant is ≤ baseline at every scale. **`skip > depth-only` is consistent and real**: at 256/1024 depth+skip beats depth-only by **+4.1 / +5.9 pt** (50.7 vs 46.6; 55.0 vs 49.0) — cross-scale exchange contributes. **residual is the closest to baseline** (within seed noise at coarse/mid scales, KO ≈ flat) but never exceeds it.

## Gate verdicts

- **Gate 1 (conv / conv-UNet) — 🔴 RED, decisive.** Both conv heads collapse to random (R@10 ≈ 0.2 = 10/5000)
  at every scale and both seeds. Treating the 1152-d embedding axis as a 1D spatial sequence destroys
  retrievability — **there is no 1D locality for convolution to exploit** (expected, confirmed strongly).
  *Honest note:* the collapse is total (not "weaker-but-trained"); conv heads weren't separately tuned, so
  read this as "conv as-applied is unsuitable," not a finely-controlled negative. Cost is also worse (2–4×
  params, 4–9× latency).

- **Gate 3 ★ (depth-wise extraction) — 🔴 RED vs baseline, with a real mechanism sub-finding.**
  Pulling coarse codes from shallow layers and fine codes from deep layers is **worse** than the baseline's
  single shared projection: fine scales (512/1024) *bottleneck* through the narrow 384-d trunk and plateau
  (~49 vs baseline's 64 at 1024). **However the mechanism the proposal bets on is real**: adding UNet **skip /
  information-exchange** (`depth+skip`) recovers **+6 pt** at 1024 over depth-only (55.0 vs 49.0), consistent
  across seeds → *cross-scale exchange genuinely contributes*. Top-down coarse→fine conditioning does **not**
  help (≈ depth-only). But even the best exchange variant stays **−9 pt below baseline** — skip only partially
  undoes the handicap that the depth decomposition itself introduces. So: **the multiscale "exchange" idea has
  a measurable effect, but the simple single-vector prefix-slicing is still strictly better** at every scale,
  at a fraction of the params/latency. Strong justification for the current design.

- **Gate 2 (coarse-to-fine residual) — 🔴 RED (inert, but the closest).** Encoding each scale from the
  residual left by coarser scales lands essentially **on top of the baseline**: Δ EN −0.4/−0.8/−1.0/−1.6
  across scales (within or near the ±0.06–0.38 seed noise at coarse/mid scales), KO ≈ flat (−0.8 worst).
  No scale meets GREEN (+1.0 fine / +1.5 coarse) — it is marginally *below* baseline everywhere. So
  residual refinement is **inert here: the shared single projection already captures what sequential
  residual coding would, and prefix-slicing is sufficient.** (It is the only structural variant that even
  matches the baseline regime, but at +1.2M params / 2.4× latency for no gain.)

## On-device cost (head identity)
Baseline is the cheapest by far: **842K params, ~1.5 ms/1k**. Every variant is larger and slower
(depth 1827K/3.3ms, depth+skip 2602K/4.1ms, conv-UNet 3947K/13ms) **and worse in R@10** → no variant
justifies its added cost. Deployment message (tiny prefix-sliced head) is reinforced, not challenged.

## Prior-work relation
- Our regime = nested codes + **per-scale independent search** + **frozen VLM** + multilingual. Residual-/
  hierarchical-quantization (RQ/HQ) and matryoshka-style methods get their coarse-to-fine gains by letting later
  stages *depend on* earlier ones at query time (codes used together) and by training the encoder. Here each
  prefix must retrieve **alone** and the backbone is **frozen**, so a scale cannot lean on others — which is
  exactly why our residual variant collapses to ≈baseline: with independent per-scale search there is no
  cross-stage dependency to exploit, and the frozen embedding's information is already linearly accessible by a
  single projection (consistent with the Gap-B finding that retrieval info is recoverable in a shared subspace).
  The depth-wise + skip result adds a mechanism note: cross-scale *exchange* does measurably help (skip > depth-
  only), but only to partially offset the capacity loss of splitting the projection across depths.

## One-paragraph summary (for "what to dig into")
Every structural variant is RED: **the current single-projection prefix-slicing baseline is the best and
cheapest head** in this controlled harness. The only positive signal is mechanistic, not deployable: UNet
**skip-style cross-scale exchange beats naive depth-wise extraction by ~6 pt**, confirming information exchange
*does* something — but not enough to overtake prefix-slicing. Recommendation: **do not pursue multiscale/UNet
heads as a retrieval-quality lever**; the finding's value is as a *negative result that validates the current
design*. If anything is worth a deeper look, it is *why* fine-scale codes prefer a single wide projection over
deep-trunk features (capacity bottleneck), not the multiscale exchange itself.

_Branch `paper-head-multiscale`. Reproduce: `REPO=$(pwd) .venv/bin/python scripts/head_multiscale_train.py` (GB10)._
