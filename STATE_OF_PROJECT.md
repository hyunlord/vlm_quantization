# STATE OF PROJECT — 1-bit Cross-Modal Hashing (vlm_quantization)

Consolidated record (safety-net snapshot). **Numbers are recorded from committed eval docs/CSVs with
their branch+SHA so they are clone-verifiable**; rows I personally re-ran on DGX this cycle are tagged
`[ran]`, older results carried from committed `web/PAPER_*.md` are tagged `[doc]`. Nothing here is
invented; unverified items are marked. See `web/README_DEPLOY.md` (deploy) and `ASSETS.md` (inventory).

## 1. System
- **Architecture**: frozen **SigLIP2-So400m** (1152-d) → **Matryoshka `NestedHashLayer`** (`Linear 1152→384 → LayerNorm → GELU → Linear 384→1024`, then per-bit `BatchNorm → L2 → sign`) → **1024-bit** codes (nested prefixes 8…1024). Image = frozen SigLIP-img (the index/anchor). Retrieval = **Hamming** over packed bits.
- **Two query paths**: (a) **server** — so400m text tower encodes the query (accurate); (b) **offline** — browser e5-small/MiniLM → `txt_h'` head → code (no server). On-device **image indexing** via MobileCLIP2 vision ONNX → `img_h'` head.
- **Deliverable surface**: browser PWA (`web/static/`, service worker) + FastAPI encode server (`web/query_server.py`, :8300). Packed index `index.bin` (50K imgs × 128 B). Deploy artifacts on DGX `web/static/{data,onnx,thumbs}/` (gitignored — regenerate per `README_DEPLOY.md`).

## 2. Verified results (with source)
**COCO 5K, T2I R@{1,5,10}, mAP@10** — `[doc paper-lowercase-refresh a3cc7b5: web/PAPER_LOWERCASE_REFRESH.md]`:
| system | EN R@1/5/10 | EN mAP@10 | KO R@1/5/10 | KO mAP@10 |
|---|---|---|---|---|
| **1-bit server (so400m+ft113, lowercased)** | 43.40/71.16/**81.24** | 55.15 | 30.68/55.72/**71.04** | 42.62 |
| SigLIP2 float ceiling (cosine) | 52.94/75.72/83.58 | 62.71 | 31.88/56.82/66.26 | 42.15 |
| naive sign (no head) | 38.42/62.76/72.62 | 48.95 | 21.74/38.28/52.18 | 30.50 |

> Lowercasing correction: so400m text tower is `do_lower_case=True`; the repo had used a case-sensitive path. Fix lifted EN R@10 **79.92→81.24** (+1.32); KO ~invariant. `[doc paper-lowercase-fix 5fd42ef / -refresh a3cc7b5]`

**Bit sweep (EN R@10)** `[doc a3cc7b5: paper/bits_extreme_lc.csv]`: 8b 14.3 · 64b 68.0 · 128b 75.5 · 256b 78.4 · 512b 79.7 · **1024b 80.6** · 2048b 81.4 · 4096b 81.1. → **1024 is the sweet spot** (2048 buys +0.4 for 2× storage). Index search @50K: 1024b ≈ 0.33 ms.

**Multilingual XM3600 (avg-36 R@10)** `[doc paper-multiling 09fb521 / paper-metaclip-master fe76d6f]`: server (so400m+ft113) **70.30** (lowercased; pre-fix 62.36); float ceiling 74.46; offline MiniLM-L12 **53.92** (best browser path, best on 30/36 langs). Offline beats server on weak-text-tower langs {de, te, th, hi}.

**Image-encoder sweep (10 encoders, on-device head-adapt)** `[doc paper-image-encoders-headadapt 6e72a07]`: anchor ceiling 81.24. **MobileCLIP2-S2 (35.8M) R@10 76.18** = sweet spot (~94% ceiling); **S0 (11.4M) 70.52** beats DINOv2-base (86.6M, SSL) 65.0 → **VL-alignment ≫ size**. SigLIP2-base (92.9M) 77.98 ≈ ceiling.

**On-device latency** `[doc paper-image-encoders-headadapt 6e72a07, perf panels]` — *device-dependent, from perf panel*: text query 50K ≈ **15 ms**; image encode MobileCLIP2-S0 ≈ **263 ms**, S2 ≈ **1066 ms** (vis+head, mobile WASM). Index search 50K ≈ 0.33 ms @1024b.

**Baselines / positioning** `[doc paper-baselines-exp 34492bb]`: our 1-bit server EN R@10 79.92 (pre-LC) / mAP@10 53.34; LSH/ITQ on same frozen features dominated by ours; float multilingual SoTA **NLLB-CLIP 86.06 avg36** (we do NOT beat multiling float SoTA — our pitch is **full-browser 128-byte deployment**, not absolute multiling SoTA). MIRFLICKR-25K CMH aux: 64b mAP@all 78.1 (matches SSAH, beats DCMH). `[doc paper-aaai-rigor 52bbc6f: paper/rigor_map_bit.csv]`

## 3. Negative-space map (the core analytical asset) — 10 gates
Method-space mapped by gating; baseline = frozen-head + InfoNCE+L2, 1024b R@10 ≈ 80.3. **+1.0pt = green bar.**
| # | gate | branch · SHA | verdict | one-line |
|---|---|---|---|---|
| 1 | loss-composition v1 (adapt-to-anchor) | paper-loss-composition · f0af984 | RED | weights/curriculum inert (≤0.2pt) |
| 2 | loss-composition v2 (from-scratch) | paper-loss-composition-v2 · 0ea0dff | RED | inert in both regimes; BN+L2 fix geometry |
| 3 | margin as predictor | paper-pillar2-margin · a13a807 | RED | mean\|z\| doesn't predict R@10 |
| 4 | Hamming predictors (neighbor/diralign) | paper-pillar2-hamming · b66e8f7 | RED | "5-way negative": cosine/parity/margin/neighbor/diralign all fail as predictors → R tracks negative-separation = what InfoNCE already does |
| 5 | **Lever A** negsep training loss | paper-lever-negsep · 99609ff | **RED** `[ran]` | hmargin widens margin (−12→−3.4b) but R@10 only re-trades top-1↔recall (R@1 +4.2/R@10 −5.7 @64b); InfoNCE already R@10-optimal |
| 6 | **Lever B** head arch (BN-free/LN/rot) | paper-lever-arch · c73c306 | **RED** `[ran]` | BN not essential (trains stably) but no benefit; aux hurts w/ or w/o BN → InfoNCE+L2 dominance, not BN |
| 7 | **Lever D** dynamic bits | paper-lever-dynbit · 2372274 | **GREEN (system)** `[ran]` | coarse-to-fine = 1024b R@10 at **15% of bit-ops**, 3.3× latency @1M; +4.7pt over uniform-128 at matched ops. *Caveats: established technique, memory unchanged, query-adaptive cascade RED* |
| 8 | rank/entropy as predictor (Part 1) | paper-rank-governs · f17febe | **RED** `[ran]` | upstream eff-dim ρ=0.38, code eff-rank ρ=0.12 (≪0.8); best non-tautological predictor = cosine ρ=0.79; Part 2 not entered per gate |
| — | **Lever C** backbone LoRA | (not run) | deferred | the only untested *method* regime; heavy (no cache); deployability conflict (encoder grows) |

**Takeaways**: InfoNCE+L2+BN is a near-optimal frozen-head recipe; no shallow loss/arch/predictor lever beats it; the one win (coarse-to-fine) is a deployment-efficiency/system result on an established technique.

## 4. Comprehensive anatomy — index (branch paper-anatomy · 3f5f764)
22 figures + 20 CSVs + `web/PAPER_ANATOMY.md` (11 observations). Scripts `scripts/anatomy_{extract,plot,de,plot_de,d_deep,plot_ddeep,f,plot_f}.py`.
- **A** backbone: `fig_anatomy_A{1..5}` (singular spectra, per-dim var, norms, pair-vs-random cosine, PCA/UMAP modality-gap) ← `anatomy_A_*.csv`
- **B** head stages: `fig_anatomy_B{1,2,3}` (per-stage var, value hist, \|z\| boundary) ← `anatomy_B_*.csv`
- **C** code space: `fig_anatomy_C{1..4}` (bit balance, bit-corr heatmap, Hamming pair-vs-wrong, grouped ablation) ← `anatomy_C_*.csv`
- **D** path comparison: `fig_anatomy_D{1,2,5,6}` (bit-flip, success/fail features, flip-vs-\|z\|, separation gap) ← `anatomy_D*.csv`
- **E** multilingual/on-device: `fig_anatomy_E{1,2,3}` (36-lang pair-Hamming↔R@10, lang UMAP, int8 flip) ← `anatomy_E_*.csv`
- **F** deeper cuts: `fig_anatomy_F{1,2}` (distill cumulative-variance/residual, per-lang code-entropy/effdim) ← `anatomy_F*.csv`

## 5. Honest limits / corrections (what was rejected)
- **Anatomy #11 downgraded**: "low text-emb eff-dim → weak language" holds only for the **extreme tail** (mi/te/quz); cross-36-lang it does **not** predict R@10 (Spearman ρ=0.38). `[paper-rank-governs f17febe]`
- **margin ρ is spurious**: mean\|z\| varies only at the 4th–5th decimal (BN+L2 pin it ≈0.0249); the −0.87 cross-lang ρ rides numerical noise — not a real predictor.
- **distillation collapses *into* the ceiling subspace** (90%-var in 76 dims vs ceiling 241; 77% energy in ceiling top-50) — a contraction, not off-manifold noise; centroid ~unchanged. `[paper-anatomy 3f5f764]`
- **We do not beat multilingual float SoTA** (NLLB 86.06 avg36); the contribution is 1-bit browser deployment + closing weak-lang gaps via offline head-adapt.
- **Lever D**: avg-bits win is *compute/latency*, **not storage** (gallery keeps 1024-bit codes); the algorithm (coarse-to-fine / Matryoshka adaptive retrieval) is established → system contribution, not a novel method.

## 6. Prior-work positioning
- **OrthoHash / HashCoder**: BN / lightweight-head / orthogonality already known → we claim **no novelty** there; our BN/loss-composition negatives are *consistent with* (cited as explanation), not novel over. `[paper-baselines-exp, web/PAPER_POSITIONING.md]`
- **CroVCA / coding-rate (anti-collapse)**: the rank-preservation method family is published → if pursued (Part 2), method is cited; our candidate contribution is the *diagnosis* (negative-space map) + cross-modal-distill-collapse / multilingual-degeneration analysis, not the loss.
- **NLLB-CLIP / AltCLIP / MetaCLIP2**: multilingual float ceilings (compared in `paper/master_table.csv`); we position as deployable 1-bit, not SoTA-multiling.

## 7. Reproduce (per result → branch)
- Main/bits/multiling/master tables: `git checkout paper-lowercase-refresh|paper-metaclip-master`; CSVs under `paper/`, methods in `web/PAPER_*.md`.
- Levers A/B/D: `paper-lever-{negsep,arch,dynbit}`; harness `scripts/lever_sweep.py` + `lever_run.sh` (head-only on `/tmp/emb_cache.pt`+`/tmp/emb_aug.pt`), `scripts/dynbit_eval.py`. Report `scripts/lever_report.py`.
- Rank Part 1: `paper-rank-governs`; `scripts/rank_predictor.py` (+ `_plot.py`).
- Anatomy: `paper-anatomy`; `scripts/anatomy_*.py` (needs `/tmp/ft_ko_113.pt`, `/tmp/distill_e5.pt`, `/tmp/txt_h_e5.pt`, `/tmp/e5_test_en.pt`, `/tmp/xm_so400m_lc.pt`; umap-learn+sklearn in venv).
- Deploy: this branch `deploy-snapshot`; `web/README_DEPLOY.md`.
- Run env: DGX `ssh dgx-spark`, repo `/home/hyunlord/github/vlm_quantization`, `.venv/bin/python` (torch 2.10, GB10).
