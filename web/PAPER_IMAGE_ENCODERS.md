# PAPER_IMAGE_ENCODERS — on-device image indexing via image-side head-adaptation (handoff)

Branch `paper-image-encoders-headadapt` (from `paper-encoders`). Image-side mirror of the text `txt_h'`
head-adaptation (Ext①): freeze a vision encoder E, train a small head `img_h'_E` that maps E's image
embeddings into the **frozen SigLIP2-So400m 1024-bit code space** (the anchor that built `index.bin`). This
quantifies **on-device image indexing**: can a phone hash a NEW photo with a small/mobile encoder into the
SAME code space as the shipped index? Anchors / index / text side UNCHANGED.

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

## Results — `paper/image_encoders_headadapt.csv`

| encoder | params | dim | family | on-device | **server EN R@10** | KO R@10 | offline EN R@10 | mixed R@10 | code-overlap | Ham/1024 | enc img/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **siglip2-base** | 92.9M | 768 | CLIP, VL+multiling | TJS | **77.98** (−3.3) | 66.84 | 70.18 | 77.78 | 0.649 | 177 | 85 |
| **mobileclip2-s2** | 35.8M | 512 | mobile CLIP | TJS/WebGPU | **76.18** (−1.8) | 64.28 | 67.04 | 77.44 | 0.623 | 184 | 29 |
| **mobileclip2-s0** | 11.4M | 512 | mobile CLIP | TJS/WebGPU | **70.52** | 59.32 | 61.78 | 74.68 | 0.565 | 36 | 36 |
| dinov2-base | 86.6M | 768 | image-only SSL | TJS | 65.00 | 53.12 | 55.68 | 71.16 | 0.501 | 222 | 81 |
| dinov2-small | 22.1M | 384 | image-only SSL | TJS | 61.38 | 49.78 | 52.70 | 69.90 | 0.475 | 231 | 93 |

(ceiling SigLIP2-So400m server EN/KO = 81.24 / 71.08; int8 MB ≈ params; "deltas" vs the 81.24 ceiling.)

## Findings (for the new §"On-device image indexing")
1. **Vision-language alignment ≫ size.** mobileclip2-**s0 (11.4M, VL) 70.5 BEATS dinov2-base (86.6M,
   image-only SSL) 65.0** — a 7.6× smaller VL-aligned encoder beats a large image-only one. The 1024-bit code
   space is *language-aligned* (it was built to match text); CLIP/SigLIP-family encoders project into it
   cleanly, image-only SSL (DINOv2) cannot, regardless of size. **This is the "which encoders work" answer.**
2. **Same-family ≈ ceiling.** siglip2-base reproduces the shipped code space to **−3.3pt** (77.98 vs 81.24):
   a phone can index new photos near-losslessly with the same family.
3. **Best on-device = mobileclip2-s2** (76.18, **35.8M, WebGPU + plhery ONNX ready**) — within −1.8pt of the
   far-larger siglip2-base at 2.6× fewer params; mobileclip2-s0 (70.5, **11.4M**) for extreme-small. (Strictly
   highest-R@10 on-device is siglip2-base 77.98, but it is 2.6× larger and not WebGPU-optimized.)
4. **Coexistence works.** Mixed gallery (half shipped SigLIP2 codes + half head-adapted E codes) stays strong
   — siglip2-base 77.78, mobileclip2-s2 77.44, even dinov2-small 69.9 — so phone-added photos retrieve fine
   alongside the shipped index.
5. **Offline-query (e5 text) tracks server-query** (~−6pt), and **code fidelity is approximate** (overlap
   0.47–0.65, Hamming 177–231/1024): img_h'_E does NOT byte-reproduce the SigLIP2 code, but the codes are
   *retrieval-equivalent enough* (R@10 within a few pt for VL encoders). It's adaptation, not bit-copying.

## On-device export + parity (the deployment artifact)
Like the text side (transformers.js e5 q8 + custom `txt_h.onnx`), only the tiny custom head needs exporting;
the vision tower runs via transformers.js/ONNX (TJS/WebGPU q8 — turnkey for siglip/dinov2/mobileclip2).
Exported `img_h'_E` fp32 ONNX (`web/export_imgh_onnx.py`), Python↔ONNX **parity verified**:
- `web/static/onnx/img_h_mobileclip2-s2.onnx` (2.3 MB) — onnxruntime vs torch max|Δ|=1.0e-7, **pack Hamming=0** ✓
- `web/static/onnx/img_h_siglip2-base.onnx` (2.7 MB) — max|Δ|=1.3e-7, **pack Hamming=0** ✓
(ONNX live on DGX `web/static/onnx/`, gitignored like `txt_h.onnx`; regenerate via `export_imgh_onnx.py --enc`.)
**Recommended on-device deploy: mobileclip2-s2** (vision via plhery/mobileclip2-onnx WebGPU + img_h_mobileclip2-s2.onnx).
Phone-encode latency = a follow-up (add image-encode timing to the perf panel).

## Honest scope notes
- **Tier-1 core delivered** (5 encoders spanning 11–93M, VL vs image-only). EVA02-B16 (Tier-2 VL ref) is
  running/appended. **OpenVision & TinyCLIP are NOT in this open_clip build's registry** (OpenVision needs a
  custom hf-hub config; TinyCLIP model name unregistered) → skipped; the 5-encoder core already spans the same
  size×family axes, so the curve is complete without them.
- Negative results (image-only SSL −15–20pt) are the contribution: they map the boundary of which encoders can
  index on-device. This demonstrates **on-device indexing feasibility**, not just compression.
- TRAIN_N=30000 subset (head-adapt converges on a subset, as in the NLLB/MetaCLIP2 experiments).
