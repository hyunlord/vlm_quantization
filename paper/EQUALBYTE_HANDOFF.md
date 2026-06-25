# Equal-byte Pareto — thesis go/die verdict

**Question (GPT's attack):** at the *same bytes/image*, does the 1-bit Hamming code dominate, or do
continuous/PQ representations beat it? If 64-d fp16 (=128 B) beats 1024-bit Hamming (=128 B), why store bits?

**Setup:** COCO-5K, T2I, diagonal gold, frozen SigLIP2 + the deployed head. Byte budget
B∈{16,32,64,128,256}. Same frozen features for all. PQ trained fairly on ~13k (train+gallery) ADC;
OPQ omitted (OPQ≥PQ, so its omission is *conservative* for the 1-bit case). No favorable bias.
Data: `paper/equalbyte_pareto.csv` / `.json`. Figures: `paper/equalbyte_figs/pareto_{EN,KO}.pdf`,
interactive `paper/equalbyte_pareto.html`. Repro: `EB_CPU=1 .venv/bin/python scripts/equalbyte_pareto.py`.

**Byte math (corrected):** 1152-d fp32 = 4608 B; 1024-bit = 128 B → **36×** compression (the paper's
"1/72" is wrong).

## Results — COCO R@10 at equal bytes (EN / KO)

| method | 16 B | 32 B | 64 B | 128 B | 256 B |
|---|---|---|---|---|---|
| **binary Hamming (ours)** | 74.0/63.4 | 77.5/67.5 | 78.8/69.7 | **79.9/71.1** | — (head caps at 1024 bit = 128 B) |
| asymmetric binary (binary gallery, cont. query) | **77.9/67.7** | **79.4/69.9** | 80.0/71.3 | 80.5/71.7 | — |
| int8 head-z (B dims) | 71.5/60.9 | 78.1/68.4 | **80.5/71.6** | 80.4/**72.1** | **80.4/72.2** |
| fp16 head-z (B/2 dims) | 59.7/50.2 | 71.6/61.1 | 77.9/68.6 | **80.5**/71.7 | 80.3/72.1 |
| int8 raw-PCA | 11.5/8.1 | 32.3/20.2 | 56.1/37.9 | 71.6/50.3 | 76.3/56.1 |
| fp16 raw-PCA | 3.5/2.2 | 11.6/8.2 | 32.4/20.3 | 56.2/37.9 | 71.5/50.2 |
| PQ (faiss, m=B×8bit) | 42.6/33.3 | 56.6/42.7 | 68.9/52.2 | 75.8/60.3 | 79.1/63.9 |

Per-byte accuracy winner (EN): 16 B asym-bin 77.9 · 32 B asym-bin 79.4 · 64 B int8 80.5 · 128 B
int8/fp16/asym 80.5 · 256 B int8 80.4. **Pure binary Hamming is the winner at no byte budget.**

**Latency** (200 q × 5 k gallery, CPU torch): binary sign-dot 31.5 ms · fp16-64d 19.1 ms · int8-128d
**4.4 ms**. (This is float sign-dot, *not* packed popcount — see honest caveat.)

## VERDICT: the "1-bit is the right representation" thesis is **mostly DEAD** (accuracy); one **partial** survivor.

1. **GPT's attack is correct.** At 128 B, fp16-64d (80.5) and asym-binary (80.5) beat 1024-bit Hamming
   (79.9). Pure binary is never the equal-byte accuracy winner.
2. **int8 of the *same* head-z prefix beats binary at every B≥32**, at identical bytes (e.g. 64 B:
   80.5 vs 78.8). The binary *alphabet* is not justified by accuracy — int8 of the matryoshka head is
   simply better per byte. Holds for EN and KO.
3. **What survives (partial):**
   - **Asymmetric binary** (binary *gallery* = our stored index, continuous query) is the Pareto front
     at low byte (16–32 B best) and ties at 128 B. So the *stored 128-byte binary index* is defensible
     **if the query is continuous** — but its search is float×sign, **not popcount** (no demonstrated
     latency win; "asymmetric for free" is false).
   - At **extreme low byte (16 B = 128-bit)** pure Hamming (74.0) is 2nd and beats fp16/int8/PQ — it
     survives only in that corner.
   - The **latency case for popcount in browser WASM is unmeasured** [VERIFY]; here the torch
     sign-dot is *slower* than int8 dot, so no latency advantage is shown.
4. **Matryoshka is the real lever, not binarization.** head-z truncations crush raw-PCA at every byte
   (128 B: int8-headz 80.4 vs int8-PCA 71.6); PQ on the raw embedding also trails head-z. The head's
   prefix concentration is what makes *any* low-byte representation work.

## Recommended reframe (honest)
Drop "1-bit codes are the right representation." The defensible thesis is **"128-byte, browser-native,
backend-free cross-modal retrieval"** where the representation should be **int8 head-z** (best per byte:
80.4/72.2 @128 B) or asymmetric-binary, and the contribution is (a) the **matryoshka head** that makes
low-byte deployment work for *any* alphabet, and (b) the full browser/on-device system. Binary Hamming
was an arbitrary alphabet choice that int8 of the same head beats at equal bytes; keep it only if a
*measured* popcount-WASM latency win (vs int8 dot) is demonstrated — currently it is not.

## Honest caveats / [VERIFY]
- Latency is torch float sign-dot, **not** packed popcount; the genuine popcount-vs-int8 comparison
  (esp. browser WASM) is **not measured** — [VERIFY] before claiming any 1-bit latency edge.
- OPQ omitted (compute); OPQ≥PQ would only widen binary's disadvantage (conservative).
- PQ on raw embedding (1152-d), m=B×8-bit, ADC; trained on ~13k. PQ on head-z not tried (head-z prefix
  truncation already dominates PQ here).
- B=256 binary N/A (head max 1024 bit). All numbers from `equalbyte_pareto.csv` (this branch).
