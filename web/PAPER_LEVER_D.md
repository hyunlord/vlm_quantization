# Lever D — Dynamic / Query-Adaptive Bit Allocation (`paper-lever-dynbit`)

**GREEN (efficiency/system), with honest caveats.** Eval-only on a trained nested head
(`/tmp/k1024_coco.pt`, bits [8..1024]); no training. COCO 5K test, T2I, EN + KO.
Harness `scripts/dynbit_eval.py`, data `paper/lever_D.csv`.

## Premise
Beating the R@10 frontier failed 7 gates (InfoNCE+L2 dominance). Lever D changes the *game*:
keep the same R@10 at **lower average bits processed** by exploiting the existing Matryoshka
nested codes. Honest comparison target = **uniform truncation** (fixed b-bit Hamming over all
gallery); a dynamic policy must beat that frontier to count.

## Policies
- **uniform b**: rank all gallery by b-bit Hamming (frontier to beat). avg_bits = b.
- **coarse2fine(b_lo,K)**: rank all gallery at b_lo, rerank top-K at 1024. avg_bits = b_lo + 1024·K/N.
- **cascade(b_lo,K,frac)**: query-adaptive — rerank only "hard" queries (small top1–top2 Hamming
  margin at b_lo); easy queries answered at b_lo. avg_bits = b_lo + frac·1024·K/N.

## Results (T2I R@10; uniform-1024 EN=79.98, KO=62.26)
**Cost-accuracy frontier (compute / avg bits processed):**

| policy | avg_bits | EN R@10 | vs uni-1024 | vs uniform@~same bits |
|---|---|---|---|---|
| uniform-128 | 128 | 75.26 | −4.72 | — |
| uniform-1024 | 1024 | 79.98 | 0.00 | — |
| **coarse2fine(128,K100)** | **148.5** | **79.98** | **+0.00** | **+4.7 vs uniform-128** |
| coarse2fine(64,K1000) | 268.8 | 79.98 | +0.00 | |
| cascade(128,K200,hard0.66) | 154.9 | 78.86 | −1.12 | (worse than coarse2fine) |

→ coarse-to-fine **matches full-1024 R@10 at 15% of the bit-ops** (148 vs 1024), or **+4.7pt over
uniform truncation at matched bit-ops**. Same result for KO (matches 62.26 at 148b). Both gate arms
met on the compute axis.

**Latency scaling (Nq=1000, uniform-1024 vs coarse2fine(128,K100), GB10):**

| gallery N | uniform-1024 | coarse2fine | speedup |
|---|---|---|---|
| 5,000 | 55.6 ms | 42.9 ms | 1.3× |
| 50,000 | 186 ms | 136 ms | 1.4× |
| 200,000 | 781 ms | 486 ms | 1.6× |
| 1,000,000 | 2472 ms | 744 ms | **3.3×** |

→ the compute win becomes a **real, scale-growing latency win** (3.3× at 1M; conservative — uses
argsort not topk). Marginal at tiny galleries (≤5K).

## VERDICT
- **coarse-to-fine reranking: GREEN** on the cost-accuracy frontier (R@10 of 1024-bit at ~15% bit-ops;
  +4.7pt over uniform at matched ops) and on latency at deployment scale (3.3× @1M).
- **query-adaptive (cascade) bit allocation: RED** — reranking only hard queries is *worse* than
  reranking everyone (easy-query low-bit answers lose recall); plain coarse-to-fine is already
  near-free and better. So *per-query bit allocation does not help*; two-stage rerank does.

## Honest caveats (no forcing)
1. **Memory: NO gain.** The gallery must store full 1024-bit codes for the rerank stage — avg-bits is a
   *compute/latency* metric, not storage. The on-device storage footprint is unchanged vs 1024-bit.
2. **Not a novel algorithm.** Coarse-to-fine / Matryoshka adaptive retrieval (Kusupati 2022) and
   two-stage ANN reranking are established. The contribution is empirical: the frontier + scale-latency
   on **1-bit multilingual cross-modal** codes, i.e. a **system** result, not a new method.
3. R@10 is **matched, not exceeded** vs full-1024 (the win is cost, not quality).

## Deployability
Pure inference-time search policy: encoder & head unchanged, codes unchanged → fully ort-web /
on-device compatible, drop-in. Fits the "portable 1-bit index" story and directly improves our
search-latency strength at scale. This is the **deployable efficiency contribution** (method+system
paragraph), strongest framed as "free 3× retrieval speedup at equal accuracy on large on-device indexes."
