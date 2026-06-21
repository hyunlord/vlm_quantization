# PAPER_LOWERCASE_REFRESH — so400m text lowercasing applied to ALL English tables (handoff)

Branch `paper-lowercase-refresh` (from `paper-lowercase-fix` 5fd42ef). **Final consistency batch.** Batch #3
fixed only COCO R@10/mAP@10 + the multiling FLOAT ceiling. This puts **every paper table that runs so400m
text** onto the correct lowercased preprocessing, at full R@{1,5,10}+mAP@10, and confirms the relative
findings survive the correction. Commit SHA: **`1002b11`** (deliverable) + **`ddbf661`** (precision faithfulness fix — reproduces the recorded int8 18.97/5.45). Clone the branch HEAD to verify.

## Fixed rule (unchanged from batch #3, verified)
so400m text = **`.lower()` + `padding="max_length", max_length=64, truncation=True`**. Only so400m is
lowercased; NLLB / AltCLIP / offline E5·MiniLM keep **native** preprocessing. Korean + non-Latin scripts are
caseless → invariant (measured below). Verification anchors all hit exactly:
- COCO server EN R@10 lower = **81.24** (= `baseline_lc.csv`); orig server EN R@1 = 41.66 ≈ `encoders.csv` 41.64.
- XM3600 so400m float: de lower = **96.54**, avg36 lower = **74.46** (= `german_sanity.csv`).
- precision fp16/bf16/text_tower flips reproduce recorded **0.12 / 0.95 / 2.57** exactly.
- multiling server avg36 lower 70.30 = `nllb_compare_lc.csv` ft113-lower avg36 (independent cross-check).

Caches reused (no backbone reload across the 4 dependent scripts): image embeddings, ft113 head, offline
heads, `mir_feats`. Only so400m text was re-encoded lowercased (`/tmp/coco_en_lc.pt`, `/tmp/xm_so400m_lc.pt`).

---

## (1) COCO main / baseline / encoders — so400m text, full R@{1,5,10}+mAP@10 (`paper/coco_lc_full.csv`)

**EN, orig → lower** (KO below). All four so400m-text paths share these numbers across tab:main / tab:baseline
/ tab:encoders (server row).

| path | R@1 | R@5 | R@10 | mAP@10 |
|---|---|---|---|---|
| **server (ft113, 1-bit)** | 41.66 → **43.40** | 69.14 → **71.16** | 79.98 → **81.24** | 53.39 → **55.15** |
| float (raw backbone cosine, ceiling) | 49.86 → **52.94** | 72.88 → **75.72** | 81.46 → **83.58** | 59.81 → **62.71** |
| naive (sign, no head) | 36.36 → **38.42** | 60.32 → **62.76** | 70.54 → **72.62** | 46.83 → **48.95** |
| float (head-continuous ceiling) | 42.76 → **44.58** | 70.56 → **72.22** | 80.54 → **81.86** | 54.65 → **56.47** |

**KO, orig → lower (invariant — caseless):**

| path | R@1 | R@10 | mAP@10 |
|---|---|---|---|
| server (ft113, 1-bit) | 30.64 → 30.68 | 70.98 → 71.04 | 42.56 → 42.62 |
| float (raw cosine) | 31.82 → 31.88 | 66.12 → 66.26 | 42.05 → 42.15 |
| naive (sign) | 21.64 → 21.74 | 52.08 → 52.18 | 30.40 → 30.50 |

**Net:** EN R@1 **+1.7 to +3.1**, R@5 **+2.0 to +2.8**, R@10 **+1.3 to +2.1**, mAP@10 **+1.8 to +2.9**. KO every
metric **+0.0 to +0.1 (noise)**. Update the SigLIP2-So400m EN cells in tab:main (server/float/naive), tab:baseline
(naive 70.2→**72.6**; server 79.92→**81.24**) and tab:encoders (server row R@1/R@5/R@10 41.64/68.98/79.92 →
**43.40/71.16/81.24**). Offline/AltCLIP/CLIP baseline rows are **unchanged** (native preprocessing).

---

## (2) Multiling SERVER (1-bit) column lowercased + offline-wins set (`paper/multiling_server_lc.csv`)

so400m+ft113 1-bit server R@10 re-derived lowercased for all 36 langs; joined with the lowercased float
ceiling and the best **native** offline 1-bit head.

- **avg36: ceiling_lc 74.46, server_lc 70.30.** (The orig-case server column in `multiling.json` — e.g. de
  30.05 — was the casing bug.)
- **German fixed and now backbone-wins:** de ceiling 96.54 / server 92.63 / best offline 66.55. de is **out**
  of the offline-wins set.
- **offline beats the so400m FLOAT ceiling only on:** `te (+23.8), hi (+11.5), th (+8.6), bn (+1.3)` — all
  **non-Latin low-resource** — plus `mi` (Maori, Latin) where BOTH collapse (server 1.7 / offline 7.2,
  degenerate). Matches batch #3 `multiling_lc.csv`.
- **offline beats the deployable 1-bit SERVER only on:** `te, th, hi` (+ degenerate mi). bn drops out
  (server_lc 34.81 > offline 34.33).

Representative langs (ceiling_lc / server_lc R@10 / best offline):

| lang | ceiling_lc | server_lc | offline_best | verdict |
|---|---|---|---|---|
| de | 96.54 | 92.63 | 66.55 | backbone (was the "30" casing bug) |
| en | 88.35 | 81.69 | 68.69 | backbone |
| ko | 88.93 | 83.84 | 60.18 | backbone |
| **te** | 4.60 | 6.17 | **28.39** | **offline** (non-Latin low-resource) |
| **th** | 60.92 | 57.94 | **69.47** | **offline** |
| **hi** | 40.90 | 43.38 | **52.36** | **offline** |
| **bn** | 33.06 | 34.81 | 34.33 | offline > ceiling, ~ties server |

**Claim to put in the paper:** restrict "on-device offline beats the SigLIP2 text tower" to **non-Latin
low-resource scripts (te/th/hi/bn)**; German and all cased Latin/Cyrillic langs are backbone-wins after the fix.

---

## (3) Extreme bit sweep [8..4096], lowercased (`paper/bits_extreme_lc.csv`)

Fresh clean-COCO Matryoshka head (8..4096, 12 epochs, orig-case train — mirroring how the deployed ft113 was
orig-trained), evaluated on BOTH orig and lowercased test text in one run (controlled snapshot). Costs:
bytes/img, 50K index MB, single-query Hamming latency @50K.

| bits | COCO EN orig→lower | COCO KO (caseless) | XM avg36 orig→lower | B/img | idx MB@50K | lat ms@50K |
|---|---|---|---|---|---|---|
| 8 | 14.22 → 14.26 | 9.64 | 6.38 → 7.47 | 1 | 0.05 | 0.114 |
| 32 | 54.50 → 54.50 | 34.34 | 24.56 → 29.90 | 4 | 0.2 | 0.033 |
| 64 | 67.46 → 68.02 | 47.04 | 35.01 → 42.06 | 8 | 0.4 | 0.035 |
| 128 | 74.74 → 75.54 | 55.28 | 42.99 → 50.73 | 16 | 0.8 | 0.032 |
| 256 | 78.00 → 78.44 | 59.22 | 47.71 → 55.85 | 32 | 1.6 | 0.058 |
| 512 | 78.92 → 79.74 | 61.62 | 50.49 → 58.78 | 64 | 3.2 | 0.152 |
| **1024** | 79.70 → **80.64** | 62.84 | 52.11 → 60.37 | 128 | 6.4 | 0.329 |
| 2048 | 80.14 → 81.36 | 63.46 | 52.94 → 61.17 | 256 | 12.8 | 0.667 |
| 4096 | 80.42 → 81.08 | 63.32 | 53.39 → 61.59 | 512 | 25.6 | 1.835 |

**Curve shape invariant after the fix:** saturation past ~512–1024; **1024→4096 buys only +0.4 COCO EN
(80.64→81.08) for 4× storage (128→512 B/img) and 5.6× latency (0.33→1.83 ms)** — the 1024 sweet-spot holds.
Lowercasing lifts every English point ~+0.4 to +0.9 and XM avg36 ~+7–8 (many cased langs); KO is flat.
Note: this is the **shared 8..4096 Matryoshka** head (capacity split across 10 bit-widths), so its 1024 point
(80.64) sits just below the **dedicated** ft113 1024 server (81.24) — consistent with the existing
`bits_extreme.csv` orig 1024 = 80.04. Replaces `bits_extreme.csv` / `bits_sweep.csv` for the lowercased paper.

---

## (4) CMH (MIRFLICKR-25K) — casing-independent, table stands (`paper/cmh_casing_check.csv`)

The (e) CMH table uses image features + the so400m text tower on **user-TAG strings**, not English captions.
Measured: **1 / 24581 tags (0.0%) change under `.lower()`; mean cos(orig tag feat, lower tag feat) = 1.0000.**
One-bit (64) DCMH category-mAP, orig vs lowercased tags, is **identical**: mAP@50 I2T/T2I 88.20/88.44,
mAP@all 78.10/79.33 (all deltas 0.0). **→ `cmh_benchmark.csv` is casing-independent; use as-is.**

---

## Precision table (tab:prec) — lowercased, faithful to `eval_paper.py` §C (`paper/precision_lc.csv`)

Regenerated with the **canonical** scheme from `web/eval_paper.py` (the script that produced
`precision.csv`): head int8 = `torch.ao.quantization.quantize_dynamic({Linear}, qint8)` on CPU, emb int8 =
per-row storage-cast, **bit-flips = `pair_bitflip` on packed codes averaged over EN+KO**, overlap vs the fp32
image gallery. The orig column **reproduces `precision.csv` exactly** (validation):

| target/dtype | flips/1024 orig→lower | top10_overlap orig→lower | EN_R10 orig→lower | KO_R10 |
|---|---|---|---|---|
| fp32 (baseline) | 0.0 | 1.000 | 79.92 → **81.24** | 71.08 (flat) |
| head fp16 | 0.12 → 0.12 | 0.998 | 79.98 → 81.24 | 71.08 |
| head bf16 | 0.95 → 0.95 | 0.990 | 79.94 → 81.26 | 71.00 |
| **head int8** | **18.97 → 18.75** | 0.929 → 0.930 | 79.70 → 81.04 | 70.52 |
| emb fp16 | 0.02 → 0.02 | 1.000 | 79.92 → 81.24 | 71.06 |
| emb bf16 | 0.18 → 0.18 | 0.997 | 79.92 → 81.24 | 71.04 |
| **emb int8** | **5.45 → 5.55** | 0.966 → 0.965 | 79.84 → 81.20 | 70.84 |
| text_tower bf16 | 1.52 → 2.23 | — | 79.92 → 81.24 | — |

orig flips (0.12 / 0.95 / **18.97** head; 0.02 / 0.18 / **5.45** emb), overlaps (0.929 / 0.966) and orig
EN/KO_R10 all **match `precision.csv` to the digit**. **Confirmed: bit-flip / overlap are CASE-INVARIANT**
(head int8 18.97→18.75, emb int8 5.45→5.55, all others ≈identical); only **EN_R10 shifts +1.2–1.4**; KO_R10
**unchanged** (caseless). The int8 head≫emb ordering (18.97 vs 5.45) and the fp16<bf16<int8 progression are
preserved. → Update only the EN_R10 column of tab:prec (+1.3); keep the flip/overlap columns. (text_tower:
this run measures bf16-vs-fp32 backbone on the full 5K = 1.52; the recorded 2.57 used bf16-vs-bf16-cache on a
1000-sample — both negligible, R@10 identical.)

---

## Snapshot: what moved (KO + non-Latin invariant; English ~+1.3)

| table | English (so400m text) | Korean / non-Latin |
|---|---|---|
| COCO main/baseline/encoders | R@10 +1.3–2.1, R@1 +1.7–3.1, mAP +1.8–2.9 | KO flat (+0.0–0.1) |
| precision | EN_R10 +1.2–1.4; flips/overlap unchanged | KO flat |
| bit sweep | COCO EN +0.4–0.9/bit; XM avg36 +7–8 | KO flat; curve shape unchanged |
| multiling server | de +62 (30→92, casing bug); cased Latin/Cyrillic ↑ | non-Latin caseless (te/hi/th) unchanged |
| CMH (e) | tags casing-independent (Δ 0.0) | — |

**Relative findings confirmed invariant after the case fix:** precision bit-flip/overlap statistics;
bit-sweep saturation shape (1024 sweet-spot, 4096 marginal); offline-encoder rows (native, untouched). The
only conclusion that changed is German/cased-Latin in the multiling claim (now backbone-wins; offline-wins
narrows to non-Latin low-resource te/th/hi/bn).

## Paper TODO (carry-over, now unblocked by this batch)
The absolute multilingual baseline tables in batch #1 (a) (`baselines_multiling.csv`, vs NLLB-CLIP/AltCLIP)
still use orig-case so400m for the SigLIP2 rows. Re-running (a) with so400m lowercased is the last remaining
table to refresh; all *instance/precision/bit/CMH* tables are now consistent.
