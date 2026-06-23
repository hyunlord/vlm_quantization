# Deep-Lever Multi-Gate — Summary (A, B, D done; C pending decision)

**Setup**: head-only on cached SigLIP2 embeddings (A/B) / eval-only on nested codes (D), mode=coco.
Baseline `bn-infonce@25`=80.88@1024 reproduces deployed 80.3. Metric COCO 5K EN/KO T2I R@10, 2 seeds (noise ≤0.3pt). GREEN = +1.0pt R@10 ≥3 bits (A/B); cost-accuracy frontier win (D).

| Lever | branch / SHA | best result | gate | one-line |
|---|---|---|---|---|
| **A** negsep loss | `paper-lever-negsep` 99609ff | hmargin +0.69 @1024 (−6.4@64) | 🔴 **RED** | InfoNCE already negsep-optimal for R@10; hmargin only re-trades top-1↔recall (R@1 +4.2/R@10 −5.7@64) |
| **B** head arch | `paper-lever-arch` c73c306 | none/ln/affine ≈0; rotation −0.5 | 🔴 **RED** | BN not essential (trains stably) but no benefit; aux hurts w/ or w/o BN → InfoNCE+L2 dominance, not BN |
| **D** dynamic bits | `paper-lever-dynbit` c1e7f3d | coarse2fine = 1024b R@10 @15% bit-ops, 3.3× latency@1M | 🟢 **GREEN** (system) | coarse-to-fine reranking dominates uniform truncation; **but** established technique, memory unchanged, query-adaptive cascade RED |
| **C** backbone LoRA | (not started) | — | ⏳ | the only remaining *method* shot — heavy (no cache, hours), deployability tradeoff |

## What's established (7→8 gates of method-space mapped)
1. **InfoNCE is R@10-optimal for negative separation.** Binary margin loss widens the realized margin (−12→−3.4 bits) yet R@10 doesn't rise — capacity-dependent precision/recall trade only.
2. **Per-bit BN is incidental.** LN/none/affine/rotation all within ~0.5pt and stable; aux losses hurt (more without BN). Geometry set by InfoNCE+L2, any normalizer preserves it.
3. **Coarse-to-fine retrieval is a free 3× speedup at equal R@10** on large on-device indexes (system win) — but per-query *bit allocation* itself doesn't help; memory unchanged; not a novel algorithm.

## Decision point — run C (LoRA) or stop?
- D-green (system) **sanctions skipping C** per the brief. But D is *efficiency*, not a new retrieval-quality *method*; C (moving the backbone) is the last untested method regime.
- **C cost**: no embedding cache (full backbone fwd), ~hours GPU; needs raw COCO images + SigLIP2 + LoRA injection. Gate bar +1.5pt.
- **C tradeoff**: if green, encoder changes → all shipped image codes must regenerate + on-device encoder grows → **conflicts with the "frozen backbone + portable 1-bit index" deployment message**.
- **If C RED too**: 9-gate complete method-space map → honest WACV/AAAI analysis+system paper (D = the deployable system contribution).

**Recommendation**: run C as a *gated cheap poke* (image-tower LoRA r=8, small steps) to close the method question — but it's your call given cost + the deployability conflict.
