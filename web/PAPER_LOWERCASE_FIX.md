# PAPER_LOWERCASE_FIX — so400m text lowercasing correction (handoff)

Branch `paper-lowercase-fix` (from `paper-nllb-bits-exp` 8fbf16e). **Correction batch.** Commit SHA:
**`86a2104884ca05f8e929851011ed49ee2ef4dce2`** (SHA-record line in the follow-up commit).

## The bug (verified, not assumed)
The SigLIP2-So400m **text tower expects lowercased input** — transformers' `SiglipTokenizer` has
**`do_lower_case=True`** by default and a `canonicalize_text()` that lowercases. Our pipeline tokenizes via
the **case-sensitive Gemma tokenizer** (`self.processor.tokenizer` in `src/data/*.py`; the paper harness
`tok(...)` likewise) and **never lowercases** → cased-script (Latin/Cyrillic) text scores were suppressed,
German catastrophically. **Fix = `.lower()` + keep `padding=max_length, max_length=64`** (dynamic padding is
worse: de 6.1; maxlen>64 → ValueError, 64-position hard cap). Non-Latin scripts (te/th/hi/bn/ja/ko/zh/ar…)
are caseless → unaffected. Offline encoders (E5/MiniLM) use their own native preprocessing → NOT lowercased.

## (i) Headline anchor audit — anchor IS affected, EN rises modestly; KO unchanged
- `emb_cache.pt` test-txt vs orig-case encode **cos 0.9999**, vs lowercased **0.9088** → the cache (and the
  79.92 anchor) was built **orig-case**. Reproduced: EN orig server 79.98 ≈ 79.92, float 81.46 ≈ 81.58,
  naive 70.54 ≈ 70.2 ✓.
- **Corrected COCO-5K server ft113: EN R@10 79.92 → 81.24 (+1.3), KO 71.08 → 71.04 (unchanged).** COCO
  captions are minimally cased (only sentence-initial caps) so the EN gain is small (+1.3), unlike heavily
  cased XM3600/German.
- **Deployment (offline e5) is NOT affected** — its text is e5 (native), not so400m: EN 74.0 → 74.24 (+0.24),
  KO unchanged. Headline deployment 73.64/64.64 stands.

## (iii) COCO so400m text, orig → lowercased (`paper/baseline_lc.csv`)
| path | EN orig → lower | KO orig → lower |
|---|---|---|
| server (ft113, 1bit) | 79.98 → **81.24** | 70.98 → 71.04 |
| float (cosine ceiling) | 81.46 → **83.58** | 66.12 → 66.26 |
| naive-sign (no head) | 70.54 → **72.62** | 52.08 → 52.18 |

EN +1.3–2.4 across paths; KO flat (caseless). Update the baseline table's SigLIP2-So400m EN values (naive
70.2 → 72.6; main/float as above).

## (ii) Multiling table re-scoped (`paper/multiling_lc.csv`) — claim narrows to non-Latin low-resource
so400m text tower (lowercased, all 36 langs from `german_sanity.csv`): **avg36 R@10 66.89 → 74.46**. Offline
MiniLM/e5 + NLLB/AltCLIP: native (unchanged). **Recomputed "offline-wins" (best offline 1-bit native >
so400m text tower lowercased): `{te +23.8, hi +11.5, th +8.6, bn +1.3, mi +5.7}`.** All are non-Latin
low-resource scripts except **mi** (Maori, Latin) where BOTH collapse (so400m 1.5 / offline 7.2 — degenerate,
both near-useless). **→ Restrict the "offline beats SigLIP2 text tower" claim to non-Latin low-resource
scripts (te/th/hi/bn).** German (the old flagship example) is OUT: lowercased so400m de 96.5 ≫ offline 66.6.

## (iv) NLLB vs ft113, fair re-compare (`paper/nllb_compare_lc.csv`) — corrects batch #2 (f)
ft113 so400m text now lowercased; NLLB+head unchanged (own tokenizer). XM3600 R@10:
| lang | ft113 orig → lower | NLLB+head | winner |
|---|---|---|---|
| de | 30.05 → **92.63** | 58.97 | ft113 (orig "30" was the casing bug) |
| en | 77.86 → 81.69 | 51.31 | ft113 |
| ko | 83.80 → 83.84 | 50.26 | ft113 |
| th | 57.46 → 57.94 | 51.78 | ft113 |
| hi | 43.38 → 43.38 | 44.13 | ~tie |
| **te** | 6.17 → 6.17 | **45.40** | **NLLB** |
| **avg36** | 62.36 → **70.30** | 51.45 | ft113 |

**The batch #2 "NLLB rides the backbone, beats ft113 on de/te" was casing-contaminated.** Corrected: NLLB+head
beats ft113 **only on Telugu (te)** (and ~ties hi); ft113-lowercased wins everywhere else and overall
(avg36 70.3 vs 51.45). → Reframe (f): NLLB backbone helps **only the single lowest-resource non-Latin script
(te)** under our light head budget; not a general win.

## (v) Offline encoder preprocessing — keep native (case-insensitive)
e5-small: EN 74.0 (native) vs 74.24 (lower); MiniLM: 78.16 vs 78.26; KO unchanged for both. Differences are
noise → **E5/MiniLM are effectively case-insensitive; keep native (do NOT force lowercase).** Model-specific
preprocessing principle holds.

## Anchor reproductions / invariants
- EN orig server 79.98 / float 81.46 / naive 70.54 ≈ batch-1 anchors (79.92 / 81.58 / 70.2) ✓
- so400m lowercased de 96.54 / avg36 74.46 = `german_sanity.csv` ✓; KO + all non-Latin unchanged ✓
- 1024-bit / CMH / bit-sweep (h)(e) untouched (no so400m text) ✓

## Paper implications
1. Headline: server COCO EN **79.92 → 81.24** (lowercased), KO 71.08 unchanged; deployment 73.64/64.64 stands.
2. Multilingual claim: restrict "offline > SigLIP2 text tower" to **non-Latin low-resource (te/th/hi/bn)**;
   drop German as an example (it was a casing artifact).
3. (f) NLLB: narrow to **te-only** advantage; ft113-lowercased otherwise wins.
4. Honest side effect: server EN ↑ ~1.3 widens the offline-vs-server EN gap slightly — reported as-is.
5. **Deployment gotcha worth a sentence:** SigLIP2's text tower must receive lowercased text; cased scripts
   (German) degrade catastrophically otherwise (de 37.6 → 96.5).
