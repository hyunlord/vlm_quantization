# PAPER_RIGOR_PILLAR3_MULTILING — §4 multilingual binary robustness (XM3600, 36 lang)

Branch `paper-aaai-rigor`, Pillar 3. Eval-only (`web/rigor_multiling.py`) on cached XM3600 SigLIP2-So400m features
(lowercased text tower). Per language, T→I retrieval over the 3600-image gallery: CONTINUOUS SigLIP2-text cosine vs
1024-bit BINARY (deployed ft113 txt_h/img_h codes). Δ = cont − bin1024 is the binarization loss.

## Result — `paper/rigor_multiling.csv` (T→I R@10; selected; full 36 in CSV)

| lang | script | cont R@10 | bin1024 R@10 | Δ | | lang | script | cont | bin | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| de | Latin | 96.54 | 92.63 | 3.91 | | ru | Cyrillic | 96.65 | 91.33 | 5.32 |
| fr | Latin | 96.06 | 91.46 | 4.60 | | ko | Hangul | 88.93 | 83.84 | 5.09 |
| es | Latin | 93.45 | 87.82 | 5.63 | | ja | Kanji/Kana | 80.65 | 74.91 | 5.74 |
| en | Latin | 88.35 | 81.69 | 6.66 | | zh | Han | 81.96 | 76.33 | 5.63 |
| pt | Latin | 93.84 | 86.62 | 7.22 | | ar | Arabic | 83.63 | 77.25 | 6.38 |
| **weak (backbone-limited):** | | | | | | th | Thai | 60.92 | 57.94 | 2.98 |
| te | Telugu | 4.60 | 6.17 | **−1.57** | | hi | Devanagari | 40.90 | 43.38 | **−2.48** |
| mi | Latin | 1.50 | 1.71 | −0.21 | | bn | Bengali | 33.06 | 34.81 | **−1.75** |
| quz | Latin | 10.17 | 8.68 | 1.49 | | sw | Latin | 21.96 | 22.01 | −0.05 |

**Summary: Latin Δ mean 4.48 · non-Latin Δ mean 3.57.**

## Findings (honest, hypothesis-correcting)
1. **Binarization is script-agnostic.** The 1024-bit binarization loss is modest (~4–7 pt for well-supported langs)
   and **non-Latin scripts are NOT disproportionately hurt** — non-Latin mean Δ (3.57) is *below* Latin mean (4.48).
   The naive "binary breaks non-Latin scripts" hypothesis is **not supported**.
2. **The failures are backbone-level, not binarization-level.** Languages that score low (te, mi, quz, sw, hi, bn, fil)
   score low at the **continuous** SigLIP2-text level too; several even have **Δ<0** (binary ≈ or > float at low
   absolute R@10 — quantization noise at the floor). These are **SigLIP2 text-tower coverage** gaps, and binarization
   is "innocent" for them. The deployable 1-bit code inherits, but does not amplify, the backbone's language coverage.
3. **KO (project focus) and other high-resource non-Latin (ru, ar, ja, zh) binarize cleanly** (Δ ≈ 5 pt, ~5% relative)
   — the 1-bit deployment preserves multilingual retrieval where the backbone is strong.

This is the Pillar-3 contribution: a per-language characterization showing 1-bit deployment **preserves multilingual
alignment uniformly across scripts**, isolating the residual weakness to the backbone text tower (actionable: improve
the encoder for {te, hi, bn, ...}, not the quantizer).

## Caveat / pending
- This is the **server** path (SigLIP2-text → txt_h). The deployed *offline* path (e5 → txt_h') multilingual transfer
  vs server-SigLIP-text is a bonus comparison that needs e5 encoding of the 36-lang XM3600 captions (not cached) —
  deferred; the core script-uniformity finding stands on the server path.

## Repro
`web/rigor_multiling.py --out paper/rigor_multiling.csv` on DGX (xm_so400m_lc.pt + ft_ko_113.pt; faiss binary search; <1 min).
