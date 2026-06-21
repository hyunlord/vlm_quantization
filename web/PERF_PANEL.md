# PERF_PANEL — on-device offline latency instrumentation (read it on the phone, no cable)

Branch `web-perf-panel` (from `paper-encoders`). **Pure instrumentation** for the fully-offline in-browser
query path. Lets you measure the real per-stage wall-clock on a physical phone (airplane mode, no remote
debugging) to replace the paper's Node-measured "~10–12 ms" and the "(v) not a physical phone" limitation.

**Guarantee:** search results are **byte-identical** with or without the panel. The panel only reads
`performance.now()` timings and renders a separate overlay; `perf.js` is *dynamic-imported only when the URL
has `?perf=1`*, so normal UX never loads it. (Verified: app.js diff is additive; `hammingTopK` call and the
query code `q` are unchanged; `encodeMs` used by the existing stats display is the same value as before.)

## What is measured (stage boundaries match the paper)
For each query, the offline path (`?perf=1`, mode = 오프라인) records:
- **encode** — query text → tokenize → multilingual-e5-small **int8** (transformers.js / ORT-Web) → 384-d emb
- **head**   — emb → `txt_h.onnx` (fp32, tanh inside) → `packBits` → 1024-bit (128-byte) code
- **search** — 50K-index Hamming top-10 (`hammingTopK`)
- **total** = encode + head + search ; plus `qlen`, `mode`, `online`, `ts`

(`search()` boundaries: encode includes tokenize; head includes packBits; search includes the top-K sort.)

## The panel (bottom-right overlay)
- Activated **only** by `?perf=1`. Shows: last query `enc / head / search / tot ms`, running **median** and
  **p90** (median emphasized — mobile tails are heavy) and **n**.
- Env line: **EP** (ORT-Web execution provider — `wasm` by default; SIMD/threads flags), `hardwareConcurrency`,
  **online** (must read `false` in airplane mode = proof of fully-offline), code bits (1024) / corpus size,
  and cold-load times once known.
- Buttons: **Reset** (zero the running stats after warm-up), **Copy** (dump the full accumulated JSON to a
  textarea — select/copy; also writes to the clipboard if `navigator.clipboard` is available).

## EP note (honest)
The session is created with no `executionProviders` option → ORT-Web default = **`wasm`**; the panel reports
that plus `ort.env.wasm.simd` / `numThreads`. **WebGPU is NOT enabled** (enabling it would change the code/
behavior, which this pure-instrumentation task must not do). To also measure WebGPU, that's a separate change.

## Cold load (reported once, separate from per-query)
`index_ms` (index.bin fetch+parse) and `encoder_ms` (e5 download + ORT sessions init) are recorded once, with
`sw_controlled` (true ⇒ served from the service-worker cache, i.e. a warm/repeat load).

## Measurement procedure (you, on the phone)
1. **Serve** this branch's `web/static/` (deployed URL, or on a laptop `cd web/static && python -m http.server
   8000`; phone on the same Wi-Fi → `http://<laptop-LAN-IP>:8000`). The backend isn't needed for offline mode.
2. On the phone open `…/?perf=1`. Let it **load once online** (this caches the app + index + e5 model + perf.js
   via the service worker). Switch mode to **오프라인(브라우저)** and run one query so the encoder finishes loading.
3. Turn **airplane mode ON** (confirm the panel's `online:false`).
4. Run **5 warm-up queries** → tap **Reset** → run **30 queries** (mix EN/KO) → tap **Copy** → send the JSON to
   yourself.
5. Paste the JSON back here → I fold the median/p90 + EP/device into the deployment § (and a small table if you
   want), replacing the Node numbers. Repeat on a 2nd device / WebGPU build if desired.

> Phone measurement is nice-to-have: if you skip it, the paper keeps the Node numbers + an honest caveat.

## Files
- `web/static/perf.js` — the panel + pure stat helpers (new; loaded only on `?perf=1`).
- `web/static/app.js` — additive timing hooks (split encode|head, record per query, capture EP/cold). No change
  to `search.js`, `sw.js`, the index, the ONNX models, or any result.

## Verification done (CC; phone is yours)
- `node --check` passes on `app.js`, `perf.js`, `search.js` (valid ESM).
- Stat helpers unit-tested (median/p90/quantile/statsFor, incl. empty/single/missing).
- **jsdom** render test passes: panel mounts, records render, EP/online/n shown, Copy → valid JSON, Reset clears.
- Diff review: additive only → top-K byte-identical with/without `?perf=1`.
