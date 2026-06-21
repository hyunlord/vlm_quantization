# PAPER_METACLIP_MASTER — final batch: residual lowercasing · MetaCLIP2 backbone · master table (handoff)

Branch `paper-metaclip-master` (from `paper-lowercase-refresh` a3cc7b5). **Final experiments batch.** Commit
SHA: **`__FILL__`**. Same fixed rule: SigLIP2-family text = `.lower()`+maxlen64; all other encoders (NLLB /
AltCLIP / MetaCLIP2 / offline E5·MiniLM) keep **native** preprocessing. KO + non-Latin caseless (measured
invariant). Anchors hit: SigLIP2-base orig reproduces backbones.csv 78.12 exactly; so400m de 96.54 / avg36
74.46 (german_sanity); MetaCLIP2 loads with `force_quick_gelu=True` (mandatory — see gotcha).

---

## (a) Residual SigLIP-family lowercasing gap (`paper/encoders_baselines_lc.csv`)

### (a1) Backbone table (tab:backbones) — SigLIP2-base lowercased; AltCLIP stays native
SigLIP2-base is the SAME SiglipTokenizer family → was still orig-case. Re-derived with text lowercased
(cached bb_siglip2-base head + image emb reused; only text re-encoded). **orig reproduces backbones.csv
exactly** (EN server 78.12, KO 56.7, flips 0.98/24.48/0.0/9.19 ✓).

| metric | orig → lower |
|---|---|
| server 1-bit EN R@10 | 78.12 → **79.00** (+0.88) |
| server 1-bit EN R@1 | 38.16 → **39.02** |
| head-cont ceiling EN R@10 | 78.62 → **80.04** (+1.42) |
| server 1-bit KO R@10 / R@1 | 56.70 / 21.14 (caseless, unchanged) |
| head flip bf16 / int8 | 0.98→0.97 / 24.48→24.21 (case-invariant) |
| emb flip bf16 / int8 | 0.0→0.0 / 9.19→8.70 (case-invariant) |
| MiniLM head-adapt EN/KO | 76.14 / 53.78 (offline, native, unchanged) |

**AltCLIP-m18 (XLM-R, case-sensitive, NOT SigLIP-family):** EN float R@10 **native 79.12 = lowercased
79.12** (COCO captions minimally cased; XLM-R robust) → **keep native** (lowercasing does not help; documented).

→ Update tab:backbones SigLIP2-base EN cells (server 78.12→79.0, R@1 38.16→39.02, ceiling 78.62→80.04);
KO + flips unchanged; AltCLIP/so400m rows unchanged.

### (a2) Multiling baseline table (batch #1 (a) `baselines_multiling.csv`) — so400m rows lowercased
Pure re-aggregation from the corrected CSVs (german_sanity = float-lower, multiling_server_lc = server-lower,
coco_lc_full = COCO). NLLB / AltCLIP / MiniLM rows **native, unchanged**.

| row | XM3600 de | en | hi | ko | te | th | **avg36** |
|---|---|---|---|---|---|---|---|
| so400m **float** R@10 orig→lower | 37.58→**96.54** | 83.36→88.35 | 40.9 (=) | 88.89→88.93 | 4.6 (=) | 60.04→60.92 | 66.89→**74.46** |
| so400m **server** 1-bit orig→lower | 30.05→**92.63** | 77.86→81.69 | 43.38 (=) | 83.8→83.84 | 6.17 (=) | 57.46→57.94 | 62.36→**70.30** |

(COCO: float EN 81.46→83.58, server EN 79.92→81.24; KO ~flat.) German was the casing bug (30→92.6);
non-Latin (hi/te) caseless → unchanged. **NLLB-CLIP float remains the multilingual ceiling** (de 94.0, te
77.4, avg36 86.06); our contribution stays "1-bit browser-deploy", not multilingual float SoTA.

---

## (B) MetaCLIP2 backbone hashing (`paper/metaclip2_hashing.csv`) — recipe rides the current SoTA multilingual backbone

MetaCLIP2-**worldwide** (ViT-H-14, the multilingual XM3600-SoTA variant) via open_clip; frozen backbone,
head-only, **same structure + budget as the NLLB experiment** (30K COCO-train pairs en+ko, clean, [8..1024],
25ep). embed=1024.

| XM3600 | de | te | th | hi | ko | en | **avg36** | COCO en/ko R@10 |
|---|---|---|---|---|---|---|---|---|
| **MetaCLIP2+head (1-bit)** | 64.26 | **35.62** | 56.35 | 37.62 | 59.73 | 55.56 | **52.01** | 71.86 / 64.88 |
| MetaCLIP2 float (ceiling) | 94.34 | 59.60 | 86.43 | 57.51 | 87.42 | 84.58 | **79.62** | 78.92 / 68.94 |

**Findings (mirror NLLB exactly):**
1. **Recipe rides MetaCLIP2.** Clean head-only (en+ko) inherits the backbone's low-resource multilinguality
   after binarization: **Telugu 1-bit 35.62 vs SigLIP2+ft113 te 6.17** (≫). MetaCLIP2 float te 59.6 vs
   SigLIP2 float te 4.6 — MetaCLIP2 is genuinely multilingual where SigLIP2's text tower collapses.
2. **Light budget is the ceiling.** MetaCLIP2+head avg36 **52.01 ≈ NLLB+head 51.45**; ~65% float retention
   (52.01/79.62), same as NLLB. Both below SigLIP2+**ft113 full recipe** (avg36 70.30) — the full recipe
   (aug+OI+RKD+KO-finetune) beats a clean head on the same-or-better backbone.
3. **Backbone generality confirmed at scale.** Contributions ②③④ + the 1-bit recipe now reproduce across
   **4 backbones / 3 families**: SigLIP2-so400m, SigLIP2-base, AltCLIP (XLM-R+CLIP), NLLB-CLIP, and MetaCLIP2.

**Sanity:** MetaCLIP2 float avg36 t2i R@10 79.62 with strong low-resource (te 59.6, de 94.3, th 86.4) is
consistent with a SoTA multilingual backbone (published numbers are i2t R@1, a different direction/metric, so
not a direct number-match — the low-resource strength pattern confirms a correct load).

**⚠ Gotcha (critical):** `metaclip2_worldwide` weights are **QuickGELU**; `open_clip.create_model_and_transforms`
on the bare `ViT-H-14-worldwide` config builds exact GELU and only **warns** — silently corrupting embeddings.
Must pass **`force_quick_gelu=True`**. (bf16 autocast used for ~3× encode speed on the GB10; harmless for
hashed retrieval.)

---

## (C) Master table (`paper/master_table.csv`) — everything on one footing

Recipe/preproc tags mandatory; **ft113-full vs clean-EN are NOT directly value-comparable** (the table is for
positioning/patterns — state this in the caption). `deployable_browser` = Y only when the QUERY ENCODER fits
in-browser (offline e5/MiniLM); server backbones encode server-side (1-bit search is browser-side regardless).

| backbone | text_path | recipe | bits | COCO en/ko R@10 | xm36 avg36 | deploy |
|---|---|---|---|---|---|---|
| SigLIP2-So400m+ft113 | server | ft113-full | 1024 | **81.24 / 71.04** | **70.30** | N |
| SigLIP2-So400m float | server | ceiling | – | 83.58 / 66.26 | 74.46 | N |
| SigLIP2-So400m naive-sign | server | no-head | 1152 | 72.62 / 52.18 | *(needs-measure)* | N |
| …+ft113 → e5-small | offline | head-adapt | 1024 | 74.0 / 66.2 | 34.07 | **Y** |
| …+ft113 → MiniLM | offline | head-adapt | 1024 | 78.16 / 65.44 | 53.92 | **Y** |
| SigLIP2-base+head | server | clean-EN | 1024 | 79.0 / 56.7 | *(needs-measure)* | N |
| AltCLIP-m18+head | server | clean-EN (native) | 1024 | 77.96 / 70.52 | *(needs-measure)* | N |
| NLLB-CLIP+head | server | clean (native) | 1024 | 66.78 / 60.62 | 51.45 | N |
| NLLB-CLIP float | server | ceiling | – | 78.68 / 71.72 | 86.06 | N |
| MetaCLIP2+head | server | clean (native) | 1024 | 71.86 / 64.88 | 52.01 | N |
| MetaCLIP2 float | server | ceiling | – | 78.92 / 68.94 | 79.62 | N |

**needs-measure (3, blanks left empty):** xm3600 avg36 for so400m naive-sign, SigLIP2-base+head, AltCLIP+head
(those experiments were COCO-only / post-hoc; XM36 not run for them).

---

## Snapshot: KO + non-Latin invariant; English ~+1
- SigLIP2-base EN +0.9 (server) / +1.4 (ceiling); KO + flips unchanged. AltCLIP native==lower.
- Multiling baseline so400m: de +59 (casing bug), avg36 float +7.6 / server +7.9; non-Latin (hi/te) caseless.
- MetaCLIP2 / NLLB / AltCLIP / offline: native (no lowercasing), unaffected.

## Net for the paper
Experiments are **complete**. The lowercase correction is now applied to every SigLIP-family table; MetaCLIP2
adds a 4th backbone (SoTA multilingual) confirming the recipe is backbone-general (rides multilinguality on
low-resource; full ft113 recipe is the strong config); the master table positions all configs. Remaining work
is [U]-items only (authors, license, on-phone latency, qualitative figures) → writing/polish.
