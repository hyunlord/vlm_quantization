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
(WASM/WebGPU), input size, embedding dim, ONNX MB, cold-load ms. Reset / Copy (JSON) reuse the text
panel's controls; `Copy` emits both the text and `image` blocks.

## `image.n=0` on phone — diagnosis & fix (ort-web verified)
First phone run logged `image.n=0, records:[], coldload:{}`. Reproduced the **real ort-web** path in
headless Chromium (the original parity was ort-**node** — different EP). Findings:
- The model is **NOT** the problem: `vis_mobileclip2-s2.onnx` runs fine in ort-web — **WASM** session
  1.1s / **run 0.5s** / out [1,512] ✓; WebGPU works too but **first run ≈ 35 s** (shader compile).
- Root causes were operational: (1) panel **defaulted to WebGPU** on phones (navigator.gpu present) →
  the 35 s first-run compile *looks frozen*; (2) the 143 MB model was fetched **only on button click**
  — if clicked after airplane mode it fetched offline and **failed silently** → n=0.

Fixes (image panel only; text/search/index/anchors untouched):
- **Default EP = WASM** (predictable ~0.5 s/img; verified). WebGPU is opt-in via `?imgep=webgpu`
  (its compile then lands in the unrecorded warmup).
- **Eager preload while online**: on panel mount (`?img=1` + `navigator.onLine`) the vision+head ONNX
  and ort-web are fetched immediately; status shows **"✓ 준비완료 · EP wasm — 이제 비행기모드 OK"**
  so the model is in memory (and SW-cached) *before* airplane mode.
- **Bundled same-origin samples** `/samples/0..7.jpg` (no remote-thumb CORS canvas taint, no file
  picker): one-tap "샘플 ×8". Missing files skip gracefully.
- **Failures are surfaced** in the status line (`❌ 로딩 실패: …`) — never a silent n=0.
Headless re-verify (fixed): eager preload → "준비완료 EP wasm"; "샘플 ×8" → **n=8, median vis 388.8 /
head 0.3 / total 389.1 ms (WASM)**, cold 1741 ms, no errors.

## Phone procedure
1. Open **`?perf=1&img=1`** **online** (WiFi — one-time 143 MB fp32 vision download). The panel
   **auto-starts caching**; wait for **"✓ 준비완료 · EP wasm — 이제 비행기모드 OK"** in the image
   section. (WebGPU instead: add `&imgep=webgpu` — expect a long first-run compile.)
2. Switch to **airplane mode** (proves on-device; panel `online:false`).
3. Tap **"샘플 ×8"** (bundled images) or **"+ 사진 인코딩"** (gallery photos, 10–20). The panel
   accumulates `vis / head / total` median·p90 — the model is already in memory, so no network needed.
4. Tap **Copy** → paste the JSON back. Headline: *"phone indexes a new photo in ≈ X ms (median, WASM)"*.

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
