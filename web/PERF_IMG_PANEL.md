# PERF_IMG_PANEL — on-device IMAGE-indexing latency (phone measurement)

Companion to the offline text-query latency panel (`?perf=1`). This adds the **image side**: how long
does it take a phone to hash a **new photo** into the shipped 1024-bit code space — entirely in-browser,
no backend. Pure instrumentation: the search/index/text path is byte-for-byte unchanged (separate module
`perf_img.js` + separate UI; loaded only with `?perf=1&img=1`).

## What it measures
Per photo, two stages (mirrors the paper's image-indexing path), plus CPU prep shown separately:
- **vis**  = MobileCLIP2-S2 vision ONNX → 512-d L2-normalized embedding  (`t_vis`)
- **head** = `img_h'_mobileclip2-s2` ONNX (tanh) + `packBits` → 1024-bit / 128-byte code  (`t_imgh`)
- **total** = vis + head  (the per-photo indexing cost; **no search** — this is indexing, not querying)
- `prep` (Canvas resize/crop/÷255) is recorded but excluded from vis/head.

Panel shows last + running **median / p90** (mobile latency has a heavy tail), plus env: EP
(WebGPU/WASM), input size, embedding dim, ONNX MB, cold-load ms. Reset / Copy (JSON) reuse the text
panel's controls; `Copy` emits both the text and `image` blocks.

## Phone procedure
1. Open the demo with **`?perf=1&img=1`** (WebGPU auto-selected if available; force with `&imgep=wasm`
   or `&imgep=webgpu`).
2. **Online, once:** tap **"+ 사진 인코딩"** (or "샘플 ×8") — this lazy-loads the vision+head ONNX
   (≈144 MB fp32, cached by the service worker) and warms up (WebGPU shader compile / WASM JIT; the
   warmup pass is **not** recorded).
3. Switch to **airplane mode** (proves on-device).
4. Tap **"+ 사진 인코딩"**, pick K photos from the gallery → each is encoded; the panel accumulates
   `vis / head / total` median·p90.
5. Tap **Copy** → paste the JSON back. Headline sentence: *"phone indexes a new photo in ≈ X ms
   (median, WebGPU)"*.

Measure both EPs by reloading with `&imgep=webgpu` then `&imgep=wasm`.

## On-device artifacts (served from `/onnx/`, gitignored like `txt_h.onnx`)
- `vis_mobileclip2-s2.onnx` (**fp32, 143.7 MB**) — the **faithful** path. Runs on WebGPU and WASM.
- `img_h_mobileclip2-s2.onnx` (2.3 MB, fp32) — the trained head.
- `vis_mobileclip2-s2.meta.json` — preprocessing (input 256, resize 256, crop 256, **mean=0 / std=1**,
  dim 512). `perf_img.js` fetches this at runtime, so the panel adapts if the export changes.
- `vis_mobileclip2-s2.int8.onnx` (37.5 MB) — **latency-only**, opt-in via `&imgvis=int8`. Dynamic int8
  quant drifts **~35 % of the 1024 bits** on this conv-heavy FastViT arch → NOT a faithful index; useful
  only to gauge the speed/size of a smaller file. (fp16 auto-convert hits a FastViT `Cast`-node type bug
  and is omitted.)
Regenerate: `web/export_vis_onnx.py` (fp32 + meta + parity) then `quant_vis.py` (int8).

## Parity (verified)
- **Python torch vs ONNX (vision):** max|Δ| = 2.4e-4; full chain `vis→img_h→packBits` Hamming = 1 / 4096.
- **JS vs Python (ort-node, same ORT core as ort-web WASM):** chain Hamming = **0 / 2048** (byte-identical
  codes) on a fixed input → confirms ort runs the FastViT vision ONNX and the in-browser code is faithful.
- The only browser-specific approximation is **Canvas vs PIL resampling** (bicubic→bilinear on resize):
  it perturbs a few near-zero bits but does not change the latency being measured. The phone numbers are
  the deliverable; code fidelity is established by the parity tests above.

## Why fp32 (and why size is fine)
int8 is unfaithful here (above) and fp16 export is blocked by a Cast-node bug, so fp32 is the correct
on-device weight. The 144 MB download is a **one-time, SW-cached** cost; the **per-photo latency**
(vis+head) — what this panel measures — is independent of file size. WebGPU is the intended on-device
execution path; WASM is reported as the CPU fallback.
