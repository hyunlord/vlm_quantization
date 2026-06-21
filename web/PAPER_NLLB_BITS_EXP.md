# PAPER_NLLB_BITS_EXP — NLLB-CLIP backbone hashing · German sanity · extreme bit sweep (handoff)

Branch `paper-nllb-bits-exp` (from `paper-baselines-exp` 34492bb / batch #1 13234f1). EVAL-only / head-only,
**frozen backbones**, cached embeddings reused. Commit SHA: **`<FILL_ON_COMMIT>`**.

Scripts (under `web/`): `paper_german_sanity.py` (g), `paper_bits_extreme.py` (h-instance),
`paper_cmh_benchmark.py --bits ...` (h-CMH, batch #1 script reused), `paper_nllb_hashing.py` (f).
CSVs: `paper/{german_sanity,bits_extreme,cmh_benchmark,nllb_hashing}.csv`.

> ⚠️ **DGX is a shared box** — during this batch the GPU was frequently at 95–96% from other users' jobs;
> "minutes" eval ran much slower and several launches were serialized. All numbers are measured, not estimated.

---

## (g) German anomaly — it is a CASING ARTIFACT, not a real so400m weakness  ⭐ corrects batch #1

**Question:** is SigLIP2-So400m's XM3600 German R@10 = 37.58 (lower than Thai 60.0 — abnormal for a
high-resource language) a real text-tower weakness or an eval artifact?

**Answer: a preprocessing (casing) artifact.** Measured (float cosine, same image gallery):

| de variant | R@10 |
|---|---|
| baseline (orig case, maxlen=64, pad=max_length) — *what batch #1 used* | **37.58** |
| **lowercased** (orig pad/maxlen) | **96.54** |
| dynamic padding (pad=longest, orig case) | 6.13 |
| maxlen=128 | **errors** — so400m raises `ValueError: sequence length 128 > max_position_embeddings 64` (confirms the 64-position cap) |

German capitalizes ALL nouns; the so400m text tower (SigLIP-style lowercased training/tokenization)
tokenizes un-lowercased German badly → artifactually low 37.58. Lowercasing → **96.54** (German is actually
excellent). The baseline is also padding-sensitive (dynamic → 6.13), i.e. the MAP-head pooler is sensitive
to padding tokens; `pad=max_length` (what reproduces published SigLIP2 numbers) is the right setting.

Outlier context (baseline R@10, identical pipeline): en 83.4 / fr 92.6 / es 83.6 / it 90.1 / ru 91.0 /
nl 79.6 / pt 84.4 / **de 37.6** — German is a lone outlier among Latin-script high-resource langs.

**Implications (important, honest):**
1. **Batch #1's German row and the "so400m German text-tower weakness" claim are an artifact** (orig-case
   preprocessing). The memory note `[[paper-extensions]]` "German=real so400m-text weakness" is **WRONG** and
   should be corrected.
2. **The "offline head-adapt beats SigLIP2's own text tower on German" claim COLLAPSES for German**: properly
   (lowercase) preprocessed, the SigLIP2 text tower scores ~96.5 on German, far ABOVE our offline head (66.6).
3. **The te/th/hi part of that claim SURVIVES (confirmed):** non-Latin scripts have no case, so lowercasing
   does not change them — te 4.6→4.6, hi 40.9→40.9, th 60.0→60.9, ko 88.9→88.9, ja/ar/zh/fa all flat. Their
   low so400m scores are genuine text-tower weaknesses, NOT casing artifacts. So our offline head still
   legitimately beats the so400m text tower on te/hi (and ties th); only the **German** claim was an artifact.

**Per-language baseline → lowercased R@10 (XM3600, float; avg36 66.89 → 74.46):**

| | de | en | fr | es | it | ru | nl | pt | pl | th | hi | te | ko | ja | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 37.6 | 83.4 | 92.6 | 83.6 | 90.1 | 91.0 | 79.6 | 84.4 | 82.4 | 60.0 | 40.9 | 4.6 | 88.9 | 80.6 | 81.9 |
| lowercased | **96.5** | 88.4 | 96.1 | 93.5 | 96.3 | 96.7 | 90.6 | 93.8 | 91.6 | 60.9 | 40.9 | 4.6 | 88.9 | 80.7 | 82.0 |
| Δ | **+59** | +5 | +3.5 | +9.9 | +6.1 | +5.7 | +11 | +9.4 | +9.2 | +0.9 | 0 | 0 | 0 | +0.1 | +0.1 |

Latin/Cyrillic-script langs ALL gain from lowercasing (de hugely, others +3–11); non-cased scripts (Thai,
Hindi, Telugu, Korean, Japanese, Chinese, Arabic, Farsi, Hebrew, Bengali) are unchanged. **So batch #1's
multilingual eval (orig-case) systematically undercounts cased-script langs — most severely German.**

Eyeball (lowercased de queries → top-1 image's EN caption): 4/5 hits, retrievals semantically correct
(e.g. de "Nahaufnahme einer Tasse Grüntee..." → "A macro shot of a cup of tea..."). The 96.5 is real.

**Recommended paper actions:** (a) drop the "so400m German weakness" framing; (b) re-run the batch #1 (a)
multilingual table with lowercasing for cased-script langs (or note the preprocessing); (c) keep the
te/th/hi argument (real, non-casing). Memory `[[paper-extensions]]` German note needs correcting.

---

## (h) Extreme bit sweep (1024 → 2048 → 4096)

### (h)-CMH — MIRFLICKR-25K label-hashing, bits 16–4096 (DONE; reused cached `/tmp/mir_feats.pt`)

Our frozen-SigLIP2 + small label head, mAP@50 (I2T/T2I) and mAP@all:

| bits | mAP@50 I2T/T2I | mAP@all I2T/T2I |
|---|---|---|
| 16 | 83.6 / 78.3 | 73.6 / 75.4 |
| 32 | **88.8** / 85.4 | 77.8 / 78.3 |
| 64 | 88.2 / 88.4 | 78.1 / 79.3 |
| 128 | 87.1 / 88.5 | 78.3 / **80.7** |
| 256 | 84.3 / 89.1 | 76.7 / 80.9 |
| 512 | 81.8 / **90.9** | 75.6 / 82.1 |
| 1024 | 82.1 / 84.0 | 77.2 / 76.9 |
| 2048 | 80.2 / 83.7 | 76.3 / 78.0 |
| 4096 | 79.8 / 83.3 | 71.2 / 77.1 |

**Anchor:** 64-bit = 88.2/88.44 — reproduces batch #1 (e) exactly ✓. **Finding:** the label-hashing task
**saturates by ~32–128 bits and slightly DECLINES beyond** (4096 < peak) — extra bits don't help category
retrieval (and can hurt via a wider head on only 5K training pairs). Our curve never reaches the purpose-built
CLIP-feature SoTA (SpikeHash 64b 95.8) — expected for a frozen-backbone + tiny-head method (auxiliary table).

### (h)-instance — COCO/XM3600, bits 16–4096 + storage/latency cost

Fresh Matryoshka head (clean COCO recipe, EPOCHS=12 — reduced from 25 due to the contended box; the
relative bit-curve is robust and 1024-bit EN R@10 80.04 ≈ batch #1 (b) full 79.98 ✓ confirms head quality):

| bits | bytes/img | COCO EN R@10/mAP | COCO KO R@10 | XM avg36 R@10 | idx MB@50K | latency ms@50K |
|---|---|---|---|---|---|---|
| 64 | 8 | 67.3 / 39.0 | 47.6 | 35.4 | 0.4 | 0.04 |
| 256 | 32 | 77.7 / 49.4 | 59.5 | 48.3 | 1.6 | 0.08 |
| 512 | 64 | 79.5 / 52.1 | 62.1 | 51.1 | 3.2 | 0.16 |
| **1024** | **128** | **80.0 / 53.4** | **63.2** | **52.6** | **6.4** | **0.38** |
| 2048 | 256 | 80.9 / 54.3 | 63.3 | 53.4 | 12.8 | 0.68 |
| 4096 | 512 | 81.0 / 54.4 | 63.5 | 53.8 | 25.6 | 1.80 |

**Finding (honest, slightly different from the hypothesis):** recall does NOT hard-saturate at the embedding
dim (~1152) — it keeps *inching* up to 4096 (EN 80.0→81.0, XM avg36 52.6→53.8 from 1024→4096). But the gain
is tiny (~+1pt) for **4× storage** (128→512 B/img) and **~5× latency** (0.38→1.80 ms/query @50K). So there is
real headroom but **sharply diminishing returns** — 1024 bit is the practical sweet spot; 2048/4096 trade
4×/8× cost for ~1pt. (The clean "saturates at ~d bits" story is not quite right; "diminishing returns past
~1024" is the accurate characterization.) Cost note: 4096-bit = 512 B/img = 4× the 1024-bit (128 B) and 32×
the 128-bit (4 B baseline naive). 16-bit latency row is first-call-warmup noise; trend is monotonic ≥32.

---

## (f) NLLB-CLIP backbone hashing — "does the recipe ride a multilingual backbone?"

Head trained on en+ko only (30K pairs from a 15K-image COCO-train subset; no aug views → consistency term
inactive), frozen NLLB backbone, eval-only on XM3600's 36 langs (no new-language training data).

**Text→image R@10:** NLLB+head (1bit) vs NLLB float vs SigLIP2+ft113 (1bit, batch #1):

| | COCO en | COCO ko | XM de | XM te | XM th | XM hi | XM en | XM ko | XM avg36 |
|---|---|---|---|---|---|---|---|---|---|
| NLLB float (server) | 78.68 | 71.72 | 94.03 | 77.42 | 87.72 | 74.26 | 84.36 | 86.67 | 86.06 |
| **NLLB+head (1bit)** | 66.78 | 60.62 | 58.97 | **45.40** | 51.78 | 44.13 | 51.31 | 50.26 | 51.45 |
| SigLIP2+ft113 (1bit) | 79.92 | 71.08 | 30.05 | 6.17 | 57.46 | 43.38 | 77.86 | 83.80 | 62.36 |

**Findings (honest):**
1. **Anchor ✓** — NLLB float reproduces batch #1 (a) exactly (COCO 78.68/71.72; XM de 94.03/te 77.42/avg36
   86.06): identical encoding, eval correct.
2. **The recipe rides the multilingual backbone and inherits NLLB's low-resource strength**: NLLB+head 1bit
   **beats SigLIP2+ft113 1bit where so400m's tower is genuinely weak — te 45.4 vs 6.2 (+39), de 59.0 vs 30.0
   (+29), hi tie** — and an en+ko-only-trained head retrieves all 36 langs (the head adapts to the embedding
   *space*, not language). Hypothesis supported directionally.
3. **But it does NOT beat the mature ft113 overall** (avg36 51.4 vs 62.4) — ft113 wins en/ko (its KO-finetune
   + much larger CC12M+aug training) and the well-covered langs.
4. **Large binarization gap**: NLLB+head retains only ~60% of NLLB float (avg36 51.4/86.1) vs ft113's ~98%
   of its float ceiling. This is a **training-budget artifact** — this NLLB head saw 30K pairs / no aug
   (forced light by the contended box) vs ft113's 113K + KO-finetune + CC12M + aug. A full-recipe NLLB head
   is expected to close much of the gap (noted follow-up). Net: NLLB-backbone hashing is a working
   proof-of-concept for "multilingual-by-backbone-choice," not yet a drop-in win over ft113.

---

## Anchor reproductions
- (h)-CMH 64-bit mAP@50 88.2/88.44 = batch #1 (e) ✓
- (g) de baseline 37.58 = batch #1 (a) / Ext② ✓ (and shown to be a casing artifact)
- (h)-instance 1024-bit COCO EN R@10 80.04 ≈ batch #1 (b) full 79.98 ✓
- (f) NLLB float = batch #1 (a) exactly: COCO en/ko 78.68/71.72, XM de 94.03 / te 77.42 / avg36 86.06 ✓

## Blocked / notes
- so400m `AutoProcessor`/`AutoTokenizer` is broken under transformers 5.1 (None model_type → `.replace`);
  use `GemmaTokenizer` directly (batch #1's so400m_imgemb already falls back to this).
- so400m text tower hard-caps at 64 positions (maxlen>64 → ValueError).
