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

## Round 2 — "nothing visible on phone" fix (mobile-emulation gated)
The numbers were rendering only in the tiny 10px panel text (off-screen / unnoticed on a phone), and
the 143 MB fp32 model loaded slowly with no visible progress. Fixes:
- **Big live readout** at the top of the image section (17px bold): after each photo it shows
  **"📸 N장 · vis X · tot Y ms"** + a sub-line **"median tot … · p90 … · n=… · <ep>"**. Loading shows
  "⏳ 모델 로딩 중…", ready shows "✅ 준비완료 — 샘플 ×8 누르세요", any throw shows "❌ …" on screen.
- **Panel widened** (340px / 94vw, `max-height:90vh; overflow:auto`) so it's legible and the buttons
  stay tappable on a phone.
- **Online measurement** (airplane mode dropped — latency is identical and it avoids the SW-precache
  race). The model is fetched on mount while online; you measure online.
- **Verified under mobile emulation** (Playwright Pixel 7, 412×839, isMobile, CPU 4× + network
  throttle): sample ×8 → big readout **"📸 8장 · vis 1591 · tot 1593 ms · median 1603 / p90 1777 /
  n=8 · wasm"**, 0 errors (DOM + screenshot). So on a throttled mobile CPU, ~1.6 s/photo (WASM); the
  real phone number is what we record.

## Phone procedure (online, no airplane)
1. Open **`?perf=1&img=1`** **online** (WiFi — one-time 144 MB fp32 vision download). The image
   section of the bottom-right panel **auto-loads the model**; wait for the big green
   **"✅ 준비완료 — 샘플 ×8 누르세요"**.
2. Tap **"샘플 ×8"** (bundled `/samples/*.jpg`, one tap) — or **"+ 사진"** for gallery photos. The
   **big readout** updates per photo with `vis / tot ms` and accumulates `median / p90 / n`.
3. Tap **Copy** → paste the JSON back. Headline: *"phone indexes a new photo in ≈ X ms (median, WASM)"*.
   (WebGPU instead: `&imgep=webgpu` — expect a multi-second first-run compile in the warmup.)

## On-device artifacts (served from `/onnx/`, gitignored like `txt_h.onnx`)
- `vis_mobileclip2-s2.onnx` (**fp32, 143.7 MB**) — the **faithful** path. Runs on WebGPU and WASM.
- `img_h_mobileclip2-s2.onnx` (2.3 MB, fp32) — the trained head.
- `vis_mobileclip2-s2.meta.json` — preprocessing (input 256, resize 256, crop 256, **mean=0 / std=1**,
  dim 512). `perf_img.js` fetches this at runtime, so the panel adapts if the export changes.
- `vis_mobileclip2-s2.int8.onnx` (37.5 MB) — **does NOT run in ort-web**: the WASM EP has no
  `ConvInteger` implementation (`ERROR_CODE 9` on session create). Kept for reference / non-web use only
  (`?imgmodel=int8` will error in the browser). fp16 was also attempted but onnxconverter saves it with
  external-data refs that ort-web can't load. **→ fp32 is the only browser-runnable variant**, hence the
  default; the 144 MB one-time download is the trade-off (shown with a visible loading state).
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
