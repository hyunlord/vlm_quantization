# IAAI/AAAI-2026 submission — `paper/iaai/`

## Build
```bash
# official AAAI pipeline (recommended; AAAI requires pdflatex)
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or single-command (auto multi-pass + bibtex):
tectonic main.tex
```
Output: `main.pdf` — **7 pages**, 0 undefined references/citations.

## Files
- `main.tex` — paper source (single file; all sections + abstract, per AAAI rule).
- `refs.bib` — bibliography (all entries web-verified; `[VERIFY]` notes mark unconfirmed author lists).
- `aaai2026.sty`, `aaai2026.bst` — **official AAAI 2026** style/bibliography files, obtained from the
  community-unified mirror of the AAAI 2026 Author Kit
  (github.com/lizhemin15/AAAI-2026-Latex-Unified; identical to aaai.org/authorkit26). Unmodified.
  The AAAI license permits redistribution for paper preparation; do **not** edit them.
  **TODO:** when the AAAI-27 Author Kit ships, swap in `aaai2027.sty/.bst`.
- `make_figs.py` — regenerates `figs/fig_{bits,signloss,multiling,gates}.pdf` from the repo CSVs.
  Run: `python make_figs.py` (needs matplotlib + the `paper/*.csv` files).
- `figs/*.pdf` — generated figures (Fig 1 system diagram is TikZ inline in `main.tex`).
- `aaai2026-unified-template.tex` — reference template (not part of the build).

## Blind mode
`\usepackage[submission]{aaai2026}` (current) = anonymous, fits both single-blind IAAI and
double-blind Main track. For camera-ready: `\usepackage{aaai2026}` and fill `\author{}`/`\affiliations{}`.

## Figure sources (all clone-verified CSVs)
| figure | source CSV |
|---|---|
| Fig 1 system diagram (TikZ) | — |
| Fig 2 bit-rate sweep | `paper/bits_extreme_lc.csv` |
| Fig 3 sign loss / asymmetric | `paper/signloss_quant.csv` |
| Fig 4 multilingual (36-lang) | `paper/multiling_server_lc.csv` |
| Fig 5 11-gate negative-space | verdicts from `STATE_OF_PROJECT.md` (clone-verified) |
