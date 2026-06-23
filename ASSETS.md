# ASSETS — inventory (branches, outputs, regeneration)

SHAs as of this snapshot. Experiment branches are **preserved** (this is an additive layer).
Numbers/details: see `STATE_OF_PROJECT.md`. Deploy: `web/README_DEPLOY.md`.

## Branches (one line + HEAD SHA + key outputs)
| branch | SHA | summary | key outputs |
|---|---|---|---|
| **deploy-snapshot** | _this_ | clean deployable PWA + docs | `web/README_DEPLOY.md`, `STATE_OF_PROJECT.md`, `ASSETS.md`, `web/static/*` |
| main | 5677fff | base | — |
| web-perf-panel | 7250632 | canonical PWA + perf panels (deploy base) | `web/static/*`, `web/query_server.py`, `web/PERF_*.md` |
| web-v3-hybrid | c0b9ae4 | hybrid PWA (server+offline) + SW/manifest | `web/static/{sw.js,manifest}`, `web/export_txth_onnx.py`, `web/WEB_HYBRID.md` |
| web-v2-headadapt | ef61fa3 | offline head-adapt spike | `web/headadapt_{train,eval}.py`, `web/V2_HEADADAPT_SPIKE.md` |
| web-v2-distill | 8a43728 | distillation spike | `web/distill_{train,eval}.py`, `web/V2_DISTILL_SPIKE.md` |
| web-v1 | d18f07a | first browser 1-bit search | `web/{build_index,query_server,search.js}`, `web/HANDOFF_v{1,2}.md` |
| paper-lowercase-refresh | a3cc7b5 | LC correction applied to ALL EN tables | `web/PAPER_LOWERCASE_REFRESH.md`, `paper/{coco_lc_full,bits_extreme_lc}.csv` |
| paper-lowercase-fix | 5fd42ef | so400m `.lower()` bug fix | `web/PAPER_LOWERCASE_FIX.md` |
| paper-metaclip-master | fe76d6f | 11-row master table (4 backbones/3 families) | `web/PAPER_METACLIP_MASTER.md`, `paper/master_table.csv` |
| paper-image-encoders-headadapt | 6e72a07 | 10-encoder on-device image head-adapt | `web/PAPER_IMAGE_ENCODERS.md`, `paper/image_encoders_headadapt.csv` |
| paper-baselines-exp | 34492bb | LSH/ITQ/FAISS + NLLB float ceiling + loss ablation + MIRFLICKR | `web/PAPER_BASELINES_EXP.md`, `paper/{baselines_multiling,ablation_losses,coco_category_map}.csv` |
| paper-multiling | 09fb521 | 36-lang diagnostic | `web/PAPER_MULTILING.md`, `paper/multiling.csv` |
| paper-nllb-bits-exp | 8fbf16e | NLLB backbone + bit sweep + German casing finding | `web/PAPER_NLLB_BITS_EXP.md` |
| paper-backbones | 9646e0f | SigLIP2-base + AltCLIP-m18 generalization | `web/PAPER_BACKBONES.md` |
| paper-encoders | a6e04c5 | text-encoder sweep + e5-prefix | `web/PAPER_ENCODERS.md`, `paper/encoders.csv` |
| paper-aaai-rigor | 52bbc6f | mAP@bit + FAISS-binary + pillars | `paper/rigor_map_bit.csv` |
| paper-review-exp | 0b62f6f | off-the-shelf baseline/seeds/buildcost | `web/PAPER_REVIEW_EXP.md` |
| paper-evals | a37d8b6 | eval corrections E/F | `web/PAPER_EVALS.md` |
| paper-qualitative-fig | f450425 | qualitative retrieval figure | `paper/fig_*` |
| **paper-loss-composition** | f0af984 | gate 1 — composition inert (adapt) | `web/PAPER_LOSS_COMPOSITION.md` |
| **paper-loss-composition-v2** | 0ea0dff | gate 2 — composition inert (scratch) | `web/PAPER_LOSS_COMPOSITION_V2.md` |
| **paper-pillar2-margin** | a13a807 | gate 3 — margin doesn't predict | `web/PAPER_PILLAR2_MARGIN.md`, `paper/pillar2_*.csv` |
| **paper-pillar2-hamming** | b66e8f7 | gate 4 — 5-way negative | `web/PAPER_PILLAR2_HAMMING.md` |
| **paper-lever-negsep** | 99609ff | gate 5 — negsep loss RED | `web/PAPER_LEVER_A.md`, `paper/lever_A.csv` |
| **paper-lever-arch** | c73c306 | gate 6 — head-arch RED | `web/PAPER_LEVER_B.md`, `paper/lever_B.csv` |
| **paper-lever-dynbit** | 2372274 | gate 7 — dynamic-bits GREEN(system) | `web/PAPER_LEVER_D.md`, `paper/{lever_D,LEVER_SUMMARY}.*` |
| **paper-rank-governs** | f17febe | gate 8 — rank/entropy predictor RED | `web/PAPER_RANK_GOVERNS.md`, `paper/rank_predictor.csv` |
| **paper-anatomy** | 3f5f764 | 22-fig anatomy | `web/PAPER_ANATOMY.md`, `paper/{anatomy_*.csv,fig_anatomy_*.pdf}` |
| improve-consistency-quality-tests | d34e7e4 | hash-quality ledger + product brief | `claudedocs/*` |

## paper/ catalog (by branch)
- **paper-anatomy**: `anatomy_A_*.csv` (svspectrum, perdim_var, norm_hist, cosine), `anatomy_B_*` (stage_perdim/hist, absz), `anatomy_C_*` (bit_balance, code_summary, hamming_dist, bit_ablation_grouped), `anatomy_D*` (bitflip, success_features, perquery, fail_overlap, Ddeep_{flip_vs_absz,gap}), `anatomy_E_*` (lang_hamming, int8_flip), `anatomy_F*` (cumvar, residual, lang_geometry); `fig_anatomy_{A1-5,B1-3,C1-4,D1,D2,D5,D6,E1-3,F1cumvar,F1residual,F2}.pdf`
- **paper-lever-***: `lever_A.csv`, `lever_B.csv`, `lever_D.csv`, `lever_all.csv`, `lever_report.txt`, `LEVER_SUMMARY.md`
- **paper-rank-governs**: `rank_predictor.csv`, `fig_rank_predictors.pdf`
- **paper-metaclip-master**: `master_table.csv` · **paper-aaai-rigor**: `rigor_map_bit.csv` · **paper-image-encoders-headadapt**: `image_encoders_headadapt.csv` · **paper-lowercase-refresh**: `coco_lc_full.csv`, `bits_extreme_lc.csv` · **paper-baselines-exp**: `baselines_multiling.csv`, `ablation_losses.csv`, `coco_category_map.csv`, `map_extension.csv` · **paper-pillar2-margin**: `pillar2_{margin_dist,flip_vs_margin,margin_predicts_R,perquery}.csv`

## web/PAPER_*.md catalog
`PAPER_LOWERCASE_{FIX,REFRESH}.md`, `PAPER_METACLIP_MASTER.md`, `PAPER_IMAGE_ENCODERS.md`, `PAPER_BASELINES_EXP.md`, `PAPER_MULTILING.md`, `PAPER_NLLB_BITS_EXP.md`, `PAPER_BACKBONES.md`, `PAPER_ENCODERS.md`, `PAPER_REVIEW_EXP.md`, `PAPER_EVALS.md`, `PAPER_LOSS_COMPOSITION{,_V2}.md`, `PAPER_PILLAR2_{MARGIN,HAMMING}.md`, `PAPER_LEVER_{A,B,D}.md`, `PAPER_RANK_GOVERNS.md`, `PAPER_ANATOMY.md`, `PAPER_POSITIONING.md` (each on its named branch).

## Gitignored deploy artifacts — disk location + regeneration
On DGX `~/github/vlm_quantization/web/static/` (excluded by `web/.gitignore`: `static/{data,thumbs,onnx}/`).
| artifact | path | regenerate |
|---|---|---|
| packed index + meta | `web/static/data/{index.bin,meta.json,index_info.json}` | `HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/build_index.py --n 50000 --seed 42 --index /tmp/demo_index.npz --image-root ~/data/coco --out web/static` |
| thumbnails (50K) | `web/static/thumbs/*.jpg` | same as above (`--reuse-thumbs` to skip) |
| text head ONNX | `web/static/onnx/txt_h.onnx` | `HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/export_txth_onnx.py` |
| vis+imgh S2 ONNX | `web/static/onnx/{vis,img_h}_mobileclip2-s2.onnx` | `.venv/bin/python web/export_vis_onnx.py` (needs `/tmp/imgh_mobileclip2-s2.pt`) |
| vis+imgh S0 ONNX | `web/static/onnx/{vis,img_h}_mobileclip2-s0.onnx` | `.venv/bin/python web/export_vis_s0.py` (needs `/tmp/imgh_mobileclip2-s0.pt`) |
| int8/fp16 vis S2 | `web/static/onnx/vis_mobileclip2-s2.{int8,fp16}.onnx` | `.venv/bin/python web/quant_vis.py` |
| img_h siglip2-base | `web/static/onnx/img_h_siglip2-base.onnx` | export variant (see `web/export_vis_onnx.py` pattern) |

Upstream checkpoints/caches (also DGX `/tmp`, not in git): `ft_ko_113.pt` (deployed head), `demo_index.npz` (corpus embeddings), `imgh_mobileclip2-s{0,2}.pt` (image heads), `distill_e5.pt`, `txt_h_e5.pt`, `e5_test_{en,ko}.pt`, `emb_cache.pt`, `emb_aug.pt`, `xm_so400m_lc.pt`, `coco_ko_test.pt`. Regenerated by training/embedding scripts on the respective experiment branches.
