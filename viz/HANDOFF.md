# viz/ — interactive anatomy (handoff)

Branch `analysis-formulation`. Companion to `docs/FORMULATION.md`. **Genuinely interactive** HTML
(Plotly via CDN); each fetches its JSON from `viz/data/`. Exploratory — captions are neutral; interpret together.

## Open
```bash
cd viz && python -m http.server 8800
# then open http://localhost:8800/   (index links all five)
```
Must be served over http:// (the pages `fetch()` JSON; `file://` is blocked by CORS). Needs internet for the Plotly CDN.

## The five views
| file | shows | controls |
|---|---|---|
| `forward_stages.html` | per-sample 1024-d distribution at each stage sliced→BN→z→\|z\| (where info collapses) | path dropdown · sample slider 0–11 · overlay-paths + stage dropdown · sign-balance readout |
| `distribution_explorer.html` | \|z\| hist · per-bit +1 balance · 64×64 bit-corr heatmap | **BatchNorm ON/OFF** · path checkboxes (overlay) · tab switch · heatmap-path dropdown |
| `sign_information_loss.html` | continuous-ranked vs Hamming-ranked top-15 (gold highlighted) + score-vs-Hamming scatter | query dropdown (✓/✗) · path · bit 64/256/1024 |
| `embedding_space.html` | precomputed 2D of SigLIP embedding / code space / 12 languages | space selector · PCA/UMAP (siglip) · legend + show/hide-all · hover |
| `retrieval_anatomy.html` | Hamming→gallery histogram w/ gold vs nearest-wrong markers + 32×32 bit-diff grid | path · query buttons (✓/✗); self-checks popcount(xor)==ham_gold (60/60) |

## Data (regenerate)
`viz/data/*.json` produced by `scripts/viz_data.py` on DGX (needs `/tmp/ft_ko_113.pt`, `/tmp/distill_e5.pt`,
`/tmp/txt_h_e5.pt`, `/tmp/e5_test_en.pt`, `/tmp/xm_so400m_lc.pt`; umap-learn+sklearn in venv):
```bash
REPO=$(pwd) .venv/bin/python scripts/viz_data.py   # writes viz/data/{forward_stages,distributions,sign_info,embedding_2d,retrieval_anatomy}.json
```
Faithfulness: forward/distributions at 1024-bit use the head's `BN_1024` (exact); sign_info/retrieval use the
head's **native per-bit BN** for 64/256/1024 (exact nested codes, not 1024-prefix approximations).
