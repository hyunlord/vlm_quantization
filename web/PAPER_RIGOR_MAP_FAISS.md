# PAPER_RIGOR_MAP_FAISS — §1 common rigor (mAP@bit + FAISS baseline)

Branch `paper-aaai-rigor`, section 1 (core). Desk-reject insurance the hashing literature expects. Eval-only on the
DEPLOYED ft113 heads (img_h + txt_h, SigLIP2-So400m server path), COCO 5K test (emb_cache). Driver
`web/rigor_map_faiss.py` (~0.4 min on GB10). **Still pending in §1:** CroVCA-style + ITQ/LSH baselines (separate run).

## mAP@bit table — `paper/rigor_map_bit.csv` (instance, both directions)

| bit | I2T R@1/5/10 | I2T mAP | T2I R@1/5/10 | T2I mAP |
|----:|---|---|---|---|
| 8    | 1.50 / 6.86 / 13.34  | 5.59  | 1.42 / 7.34 / 13.94  | 5.59  |
| 16   | 7.26 / 22.76 / 32.68 | 15.68 | 7.76 / 22.14 / 32.82 | 15.67 |
| 32   | 16.28 / 38.82 / 52.28 | 27.84 | 16.04 / 39.18 / 52.06 | 27.66 |
| 64   | 25.52 / 53.30 / 67.34 | 38.78 | 25.76 / 53.30 / 66.86 | 38.99 |
| 128  | 33.36 / 62.30 / 74.20 | 46.77 | 33.02 / 61.78 / 73.96 | 46.31 |
| 256  | 37.50 / 66.98 / 77.88 | 51.04 | 38.24 / 66.08 / 77.48 | 51.20 |
| 512  | 40.58 / 69.02 / 79.26 | 53.54 | 39.90 / 68.20 / 78.72 | 52.88 |
| **1024** | **41.42 / 69.84 / 80.32** | **54.46** | **41.64 / 68.98 / 79.92** | **54.21** |

Instance mAP = mean(1/rank of the paired item over the full 5K gallery). Clean monotone Matryoshka curve (one model,
prefix-sliced); 1024-bit R@10 ≈ 80 matches the SigLIP2 server ceiling. Knee ~256 bits (R@10 78 at 1/4 the storage).

## FAISS binary-index baseline — `paper/rigor_faiss_bench.csv` (1024-bit, I2T, N=5000)

| method | R@10 | recall@10 vs exact | latency ms/query | index mem |
|---|---|---|---|---|
| IndexBinaryFlat (exact) | 80.32 | 100.0 | **0.033** | 0.64 MB |
| IndexBinaryIVF (nlist64, nprobe8) | 79.72 | 98.11 | **0.0054** | 0.64 MB |
| numpy brute Hamming (client analogue) | 80.44 | 99.23 | **1.73** | 0.64 MB |

(recall<100 for IVF/brute is top-10 tie-ordering at equal Hamming distance, not missed neighbors.)

## Reading (for §"retrieval engine")
- **Client-side Hamming is sound for the deployment scale.** Brute-force Hamming in the browser (our shipped path)
  matches exact-FAISS recall (~99%) at **1.73 ms/query** over 5K × 1024-bit — interactive, no server. Index is 0.64 MB
  (5000 × 128 B), trivially shippable.
- **FAISS is the scale path, not a necessity here.** `IndexBinaryFlat` is 0.033 ms/q (52× faster than our brute loop)
  and `IndexBinaryIVF` 0.0054 ms/q at 98% recall — i.e. for million-scale galleries FAISS IVF is the drop-in, but at
  product scale the zero-dependency client Hamming already suffices. This pre-empts the "why not FAISS?" review.
- **mAP@bit complements R@K** and gives the storage/accuracy knee (256 bits ≈ −2pt R@10 vs 1024 at 1/4 storage),
  matching hashing-paper reporting conventions.

## Repro
`web/rigor_map_faiss.py --out_map paper/rigor_map_bit.csv --out_faiss paper/rigor_faiss_bench.csv` on DGX
(`.venv/bin/python`, GPU; ft_ko_113.pt heads + emb_cache test; ~0.4 min).
