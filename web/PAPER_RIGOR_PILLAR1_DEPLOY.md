# PAPER_RIGOR_PILLAR1_DEPLOY — §2 on-device binary deployment science

Branch `paper-aaai-rigor`, Pillar 1. Two parts: (A) precision/bit-flip sensitivity (eval, `web/rigor_bitflip.py`),
(B) runtime-constraints table (browser deployment facts + measured phone latency). The honest headline corrects a
naive assumption: **int8 precision is numerically harmless; the deployment blocker is operator/format support, not
precision.**

## A. Precision / bit-flip sensitivity — `paper/rigor_bitflip.csv` (COCO 5K, I2T, fp32 ref R@10 = 80.32)

| quant point | precision | bit-flip% img | bit-flip% txt | R@10 | R@10 drop |
|---|---|---|---|---|---|
| encoder-output | fp32 | 0.000 | 0.000 | 80.32 | 0.00 |
| encoder-output | fp16 | 0.004 | 0.003 | 80.36 | −0.04 |
| encoder-output | int8 | 1.014 | 0.902 | 80.38 | −0.06 |
| head-output | fp32 | 0.000 | 0.000 | 80.32 | 0.00 |
| head-output | fp16 | 0.000 | 0.000 | 80.32 | 0.00 |
| head-output | int8 | 0.817 | 0.816 | 80.38 | −0.06 |

int8 = symmetric per-tensor (scale = max|x|/127); fp16 = round-trip half.

**Finding (honest, hypothesis-correcting):** quantizing to int8 — at *either* the encoder output (input feature) or
the head output (pre-sign z) — flips only ~1% of bits and costs **0 R@10** (drops are within tie-noise). So the
"head int8 is numerically critical" hypothesis is **not supported**: at the math level, int8 precision is fine for
this code. The pre-sign margins are tiny (≈0.025) yet only ~0.8% of bits sit close enough to zero to flip under int8,
and those flips don't change top-10 retrieval. fp16 is essentially lossless everywhere.

## B. Runtime constraints — why deployment is fp32-only anyway (the real blocker)

| variant | runs in ORT-Web (WASM)? | why | size (vis encoder) |
|---|---|---|---|
| **fp32** | **yes** | all ops supported | MobileCLIP2-S0 46 MB / S2 144 MB |
| fp16 | **no** | onnxconverter saves external-data weight refs ort-web can't load | ~half |
| int8 | **no** | WASM EP has no `ConvInteger` impl → `ERROR_CODE 9` on session create | ~¼ (≈params) |

EP / latency (measured, MobileCLIP2 vision + head, per photo, mobile-emulation Pixel 7):
- **WASM fp32**: predictable; **S0 ≈ 263 ms, S2 ≈ 1066 ms** per photo (vis+head). Default EP.
- **WebGPU**: intended on-device path; first-run shader compile ~35 s (lands in warmup), then fast.

**Finding:** the §A result shows int8/fp16 would be *numerically* fine (≤1% flip, 0 R@K loss), so the move to
**fp32-only on-device is forced by the runtime, not by accuracy** — ort-web's WASM EP cannot execute `ConvInteger`
(int8) and cannot load external-data fp16. This *strengthens* the deployment-science point: encoder choice and the
one-time fp32 download are dictated by browser operator/format support, and the resulting per-photo latency
(263–1066 ms by encoder) is what bounds the on-device indexing UX — independent of code precision.

## Figures (to render from CSV in the writing pass)
- bit-flip% vs precision (encoder-output vs head-output) — flat-low, the "precision is cheap" panel.
- phone latency vs encoder/model size (S0/S2, WASM) — the "runtime bounds UX" panel.

## Repro
`web/rigor_bitflip.py --out paper/rigor_bitflip.csv` on DGX (ft_ko_113.pt + emb_cache test; <1 min). Latency/runtime
facts from the perf panel (`web-perf-panel`, `?perf=1&img=1`) and PERF_IMG_PANEL.md.
