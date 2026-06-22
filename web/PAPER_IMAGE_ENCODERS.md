# PAPER_IMAGE_ENCODERS — on-device image indexing via image-side head-adaptation (handoff)

Branch `paper-image-encoders-headadapt` (from `paper-encoders`). Image-side mirror of the text `txt_h'`
head-adaptation (Ext①): freeze a vision encoder E, train a small head `img_h'_E` that maps E's image
embeddings into the **frozen SigLIP2-So400m 1024-bit code space** (the anchor that built `index.bin`). This
quantifies **on-device image indexing**: can a phone hash a NEW photo with a small/mobile encoder into the
SAME code space as the shipped index? Anchors / index / text side UNCHANGED. Curve-extension commit SHA:
**`38963ca`** (10 encoders; original 5-core Tier-1 in `ac56101`).

## Method (mirror of headadapt_train.py)
- Anchor = `ft113 img_h(SigLIP2 image emb)` (frozen — the exact head behind `index.bin`).
- Per encoder E (frozen): encode COCO-train subset (TRAIN_N=30000) + 5K test with E → train
  `img_h'_E = NestedHashLayer(E_dim→hidden→1024)` with `CombinedHashLoss(io=frozen anchor codes,
  to=img_h'_E(E_emb))`, 25 epochs, hp_results recipe. The InfoNCE pairing (same image i) makes E-codes
  reproduce the SigLIP2 anchor code per image; text-query searchability follows transitively (text codes
  already align to SigLIP2 image codes). Driver: `web/paper_image_headadapt.py --enc <name>`.
- Eval COCO 5K test EN/KO: **(a)** E-indexed gallery searched by server SigLIP2-text (lowercased) and offline
  e5+txt_h' queries → R@{1,5,10}; **(b)** mixed gallery (half shipped SigLIP2 codes + half E codes) → R@10
  (phone photos coexisting with the shipped index); **(c)** code fidelity = per-image Hamming(E,SigLIP2) +
  top-10 overlap. Ceiling = the shipped SigLIP2-So400m server (EN 81.24 / KO 71.08).

## Results — `paper/image_encoders_headadapt.csv` (10 encoders, sorted by params)

| encoder | params | dim | family | on-device | **server EN R@10** | KO R@10 | offline EN R@10 | mixed R@10 | overlap | Ham/1024 |
|---|---|---|---|---|---|---|---|---|---|---|
| **openvision-ti16** | 5.6M | 192 | CLIP VL | export | 61.20 | 50.66 | 51.44 | 68.28 | 0.460 | 234.8 |
| **tinyclip-8m** | 8.3M | 512 | CLIP VL (distill) | export | 61.60 | 50.68 | 53.24 | 69.10 | 0.476 | 227.0 |
| **mobileclip2-s0** | 11.4M | 512 | mobile CLIP | TJS/WebGPU | 70.52 | 59.32 | 61.78 | 74.68 | 0.565 | 202.3 |
| **openvision-s16** | 21.8M | 384 | CLIP VL | export | 71.38 | 59.40 | 62.78 | 74.84 | 0.560 | 204.6 |
| dinov2-small | 22.1M | 384 | image-only SSL | TJS | 61.38 | 49.78 | 52.70 | 69.90 | 0.475 | 230.9 |
| **mobileclip2-s2** | 35.8M | 512 | mobile CLIP | TJS/WebGPU | **76.18** | 64.28 | 67.04 | 77.44 | 0.623 | 184.0 |
| dinov2-base | 86.6M | 768 | image-only SSL | TJS | 65.00 | 53.12 | 55.68 | 71.16 | 0.501 | 222.1 |
| siglip2-base | 92.9M | 768 | CLIP VL+multiling | TJS | 77.98 | 66.84 | 70.18 | 77.78 | 0.649 | 176.7 |
| pe-core-b16 | 93.7M | 1024 | VL (Meta PE) | export | 76.14 | 64.30 | 67.90 | 76.72 | 0.626 | 183.4 |
| mobileclip2-s4 | 321.8M | 768 | mobile CLIP | export | 77.62 | 66.72 | 69.86 | 77.30 | 0.658 | 172.5 |

(ceiling SigLIP2-So400m server EN/KO = 81.24 / 71.08; int8 MB ≈ params; deltas below are vs the 81.24 ceiling.)

## Findings (for the new §"On-device image indexing")
1. **Vision-language alignment ≫ size — the dominant axis.** mobileclip2-**s0 (11.4M, VL) 70.5 BEATS
   dinov2-base (86.6M, image-only SSL) 65.0** (7.6× smaller, +5.5pt). At the extreme-small end, VL encoders
   (openvision-ti16 5.6M=61.2, tinyclip-8m 8.3M=61.6) **match dinov2-small (22.1M SSL) 61.4 at 3–4× fewer
   params**. The 1024-bit code space is *language-aligned*; CLIP/SigLIP-family encoders project into it,
   image-only SSL cannot, regardless of size. **This is the "which encoders work" answer.**
2. **Mobile-distilled VL is the most param-efficient.** MobileCLIP2 dominates per-param: s2 (35.8M) **76.2**
   ≈ pe-core-b16 (93.7M, 76.1) and matches siglip2-base-class accuracy at a fraction of the cost.
3. **Returns flatten past ~36M.** s2 (35.8M) 76.2 ≈ pe-core-b16 (93.7M) 76.1; mobileclip2-s4 (321.8M) 77.6 ≈
   siglip2-base (92.9M) 78.0 — i.e. 9× more params buys ≤+1.5pt. The knee is ~36M → **mobileclip2-s2 is the
   sweet spot**; the residual gap to the 81.24 ceiling is a family/scale limit, not a size limit.
4. **Same-family ≈ ceiling.** siglip2-base reproduces the shipped code space to −3.3pt (77.98 vs 81.24): a
   phone can index new photos near-losslessly with the same family.
5. **Coexistence + approximate fidelity.** Mixed gallery (half shipped SigLIP2 + half head-adapted E) stays
   strong for every encoder (68.3–77.8), so phone-added photos retrieve fine alongside the shipped index.
   Code fidelity is *approximate* (overlap 0.46–0.66, Hamming 173–235/1024) and tracks VL-alignment — it's
   retrieval-equivalent adaptation, not bit-copying. Offline-query (e5 text) tracks server-query (~−6–10pt).

## Best on-device → export + parity (the deployment artifact)
**Recommended on-device deploy = mobileclip2-s2** (76.18, **35.8M, WebGPU**; sweet-spot per finding 3).
Like the text side (transformers.js e5 q8 + custom `txt_h.onnx`), only the tiny custom head needs exporting;
the vision tower runs via ONNX (WebGPU/WASM). Exported & parity-verified:
- `img_h'` fp32 ONNX (`web/export_imgh_onnx.py`): `img_h_mobileclip2-s2.onnx` (2.3 MB), pack Hamming=0, max|Δ|~1e-7.
- **Vision tower** MobileCLIP2-S2 → ONNX (`web/export_vis_onnx.py`, exported myself — plhery/mobileclip2-onnx
  turned out to be a mis-committed venv, unusable): `vis_mobileclip2-s2.onnx` (**fp32 143.7 MB, the faithful
  path**; input 256², mean=0/std=1, dim 512). torch-vs-ORT max|Δ|=2.4e-4, full chain Hamming=1/4096; **JS
  (ort-node, same core as ort-web wasm) vs python chain Hamming=0/2048 — byte-identical**. int8 dynamic quant
  drifts ~35% of bits on this conv-heavy FastViT arch (latency-only); fp16 auto-convert hits a Cast-node bug.
  (ONNX live on DGX `web/static/onnx/`, gitignored like `txt_h.onnx`.)
Phone image-encode latency is measured by the Task-B perf panel (branch `web-perf-panel`, `?perf=1&img=1`).

## Honest scope notes (loaders / gotchas)
- **All 10 candidates loaded** — the earlier "loading failed" set is resolved: **OpenVision** Ti/16 (5.6M) &
  S/16 (21.8M) load via open_clip `hf-hub:UCSC-VLAA/...`; **TinyCLIP-8M** loads via **transformers
  `CLIPModel`** (the wkcn repo is HF-CLIP format, NOT open_clip) using `vision_model.pooler_output →
  visual_projection` (its `get_image_features` returns a ModelOutput, not a tensor — the `.float()` bug);
  **PE-Core-B-16** via open_clip `meta` (EVA02 substitute — EVA02 is GB10-slow); **MobileCLIP2-S4** (321.8M)
  finished within a 35-min timebox.
- Negative results (image-only SSL −15–20pt) are the contribution: they map the boundary of which encoders
  can index on-device. Demonstrates **on-device indexing feasibility**, not just compression.
- TRAIN_N=30000 subset (head-adapt converges on a subset, as in the NLLB/MetaCLIP2 experiments).
