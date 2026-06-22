/* On-device IMAGE-indexing latency — PURE INSTRUMENTATION, loaded ONLY with ?perf=1&img=1.
 * Measures the cost to hash a NEW photo into the shipped 1024-bit code space, entirely in-browser:
 *   vis  = MobileCLIP2-S2 vision ONNX  (encode image -> 512-d L2 emb)   [t_vis]
 *   head = img_h'_mobileclip2-s2 ONNX (tanh) + packBits -> 1024-bit code [t_imgh]
 *   total = vis + imgh  (NO search — this is INDEXING a photo, not querying)
 * The vision ONNX bakes L2-normalize into its output, so head input is turnkey. Preprocessing
 * (resize shorter side -> center-crop -> /255; MobileCLIP2 uses mean=0/std=1) replicates the
 * Python open_clip transform; minor Canvas-vs-PIL resampling differences are inherent (noted in
 * PERF_IMG_PANEL.md). Search results / text path are completely untouched (separate module + UI).
 */
import { packBits } from "./search.js";

const CDN_ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.mjs";
// Model registry (all fp32 — the ONLY browser-runnable dtype: ort-web WASM has no ConvInteger for
// int8, and onnxconverter fp16 uses external-data ort-web can't load). DEFAULT = s0 (~46MB): 3×
// lighter than s2 → fast mobile load + fits phone WASM memory. ?imgmodel=s2 → bigger/faithful 144MB.
const MODELS = {
  s0: { vis: "/onnx/vis_mobileclip2-s0.onnx", head: "/onnx/img_h_mobileclip2-s0.onnx", meta: "/onnx/vis_mobileclip2-s0.meta.json", mb: 46, label: "MobileCLIP2-S0" },
  s2: { vis: "/onnx/vis_mobileclip2-s2.onnx", head: "/onnx/img_h_mobileclip2-s2.onnx", meta: "/onnx/vis_mobileclip2-s2.meta.json", mb: 144, label: "MobileCLIP2-S2" },
};
const MODEL = MODELS[new URLSearchParams(location.search).get("imgmodel")] || MODELS.s0;

let _ort = null, _vis = null, _head = null, _meta = null, _ep = null;

function pickEP() {
  // Default WASM: predictable per-image latency (~0.5s/img desktop; verified in headless ort-web,
  // session 1.1s / run 0.5s). WebGPU's FIRST run pays a multi-second shader-compile (≈35s observed
  // headless) that looks frozen on a phone — opt in with ?imgep=webgpu (compile then lands in the
  // unrecorded warmup). WASM was the auto-fallback that actually worked, so make it the default.
  return new URLSearchParams(location.search).get("imgep") === "webgpu" ? "webgpu" : "wasm";
}

async function head(url) {  // does a vision file exist? (HEAD; fall back to fp32 if not)
  try { return (await fetch(url, { method: "HEAD" })).ok; } catch (_) { return false; }
}

async function load(PERF, setStatus) {
  if (_vis) return;
  const t0 = performance.now();
  setStatus("이미지 인코더 로딩… (vision + head ONNX, 최초 1회)");
  _ort = await import(CDN_ORT);
  _meta = await (await fetch(MODEL.meta)).json().catch(() => ({ input: 256, resize: 256, crop: 256, dim: 512 }));
  _ep = pickEP();

  const providers = _ep === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  setStatus(`vision 모델 로딩중 (${MODEL.label} fp32 ~${MODEL.mb}MB)… 최초 1회, 잠시만요`);
  _vis = await _ort.InferenceSession.create(MODEL.vis, { executionProviders: providers });
  _head = await _ort.InferenceSession.create(MODEL.head, { executionProviders: ["wasm"] });

  // warmup (WebGPU shader compile / wasm JIT) — not recorded as a real measurement
  const warm = new _ort.Tensor("float32", new Float32Array(3 * _meta.input * _meta.input), [1, 3, _meta.input, _meta.input]);
  const we = (await _vis.run({ px: warm }))[_vis.outputNames[0]];
  await _head.run({ emb: we });

  const onnxMb = await fetch(MODEL.vis).then((r) => +(r.headers.get("Content-Length") || 0) / 1e6).catch(() => null);
  PERF.setImgEnv({ ep: _ep, model: MODEL.label, input: _meta.input, dim: _meta.dim,
    onnx_mb: onnxMb ? Math.round(onnxMb) : null, vis_file: MODEL.vis.split("/").pop(),
    faithful: MODEL === MODELS.s2 });
  PERF.setImgCold({ load_ms: +(performance.now() - t0).toFixed(1),
    sw_controlled: !!(navigator.serviceWorker && navigator.serviceWorker.controller) });
  setStatus(`<span style="color:#36d399">✓ 준비완료 · ${MODEL.label}/${_ep}</span> — "샘플 ×8"을 누르세요`);
}

/* ---- preprocessing: replicate open_clip Resize(shorter->R, bicubic) + CenterCrop(C) + /255 ---- */
function preprocess(img) {
  const R = _meta.resize || _meta.input, C = _meta.crop || _meta.input;
  const w = img.width, h = img.height;
  const scale = R / Math.min(w, h);
  const rw = Math.round(w * scale), rh = Math.round(h * scale);
  const cv = document.createElement("canvas"); cv.width = C; cv.height = C;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  // center-crop: draw the resized image so its center lands in the CxC canvas
  const dx = (C - rw) / 2, dy = (C - rh) / 2;
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, dx, dy, rw, rh);
  const { data } = ctx.getImageData(0, 0, C, C);     // RGBA, [0,255]
  const chw = new Float32Array(3 * C * C);           // mean=0 std=1 -> just /255
  const plane = C * C;
  for (let i = 0; i < plane; i++) {
    chw[i] = data[i * 4] / 255;
    chw[plane + i] = data[i * 4 + 1] / 255;
    chw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return new _ort.Tensor("float32", chw, [1, 3, C, C]);
}

async function encodeOne(img, PERF) {
  const t0 = performance.now();
  const px = preprocess(img);                          // CPU prep (not counted in vis/imgh)
  const tA = performance.now();
  const emb = (await _vis.run({ px }))[_vis.outputNames[0]];
  const tB = performance.now();
  const cont = (await _head.run({ emb }))[_head.outputNames[0]].data;
  const code = packBits(cont);                          // 128 bytes
  const tC = performance.now();
  PERF.recordImg({ vis: +(tB - tA).toFixed(2), imgh: +(tC - tB).toFixed(2),
    total: +(tC - tA).toFixed(2), prep: +(tA - t0).toFixed(2), ep: _ep, bytes: code.length });
  return code;
}

function fileToImage(file) {
  return new Promise((res, rej) => {
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = () => { URL.revokeObjectURL(url); res(im); };
    im.onerror = (e) => { URL.revokeObjectURL(url); rej(e); };
    im.src = url;
  });
}

const SAMPLE_DIR = "/samples";   // bundled SAME-ORIGIN JPEGs (no remote-thumb CORS canvas taint)
const SAMPLE_N = 8;

function loadImageURL(src) {       // same-origin → no crossOrigin needed, canvas not tainted
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("img load failed: " + src));
    im.src = src;
  });
}

function fmt(v) { return v == null ? "–" : Math.round(v); }

export async function initImgPanel(PERF) {
  const box = PERF.imgContainer();
  if (!box) return;
  // widen the panel so the big readout is legible on a phone
  const panel = document.getElementById("perfPanel");
  if (panel) { panel.style.width = "340px"; panel.style.maxWidth = "94vw";
    panel.style.maxHeight = "90vh"; panel.style.overflowY = "auto"; }  // scrollable so buttons stay tappable on phones

  box.innerHTML =
    '<input id="pfImgFile" type="file" accept="image/*" multiple style="display:none">' +
    // BIG live readout — impossible to miss
    '<div id="pfImgBig" style="margin:2px 0 6px;padding:8px;border-radius:8px;background:#0b0d12;border:1px solid #2a3247;text-align:center">' +
    '<div id="pfImgBigMain" style="font-size:17px;font-weight:800;color:#36d399;line-height:1.25">📸 이미지 인코딩</div>' +
    '<div id="pfImgBigSub" style="font-size:11px;color:#8b94a7;margin-top:2px">모델 준비 중…</div>' +
    "</div>" +
    '<div style="display:flex;gap:6px">' +
    '<button id="pfImgPick" style="flex:1;font:inherit;padding:7px;border:1px solid #2a3247;border-radius:6px;background:#141821;color:#e7ebf3;cursor:pointer">+ 사진</button>' +
    '<button id="pfImgSample" style="flex:1;font:inherit;padding:7px;border:1px solid #5b9dff;border-radius:6px;background:#16233b;color:#cfe0ff;cursor:pointer;font-weight:700">샘플 ×8</button>' +
    "</div><div id='pfImgStatus' style='margin-top:4px;color:#8b94a7;font-size:10px'></div>";
  const status = (h) => { const s = box.querySelector("#pfImgStatus"); if (s) s.innerHTML = h; };
  const big = (main, sub, color) => {
    const m = box.querySelector("#pfImgBigMain"), s = box.querySelector("#pfImgBigSub");
    if (m) { m.innerHTML = main; if (color) m.style.color = color; }
    if (s && sub != null) s.innerHTML = sub;
  };
  const showStats = () => {
    const im = PERF.snapshot().image, st = im.stats, last = im.records[im.records.length - 1];
    if (!last) return;
    big(`📸 ${im.n}장 · vis <b>${fmt(last.vis)}</b> · tot <b>${fmt(last.total)}</b> ms`,
        `median tot <b>${fmt(st.total.median)}</b> · p90 <b>${fmt(st.total.p90)}</b> · n=${im.n} · ${im.env.ep || "?"}`,
        "#36d399");
  };
  const file = box.querySelector("#pfImgFile");

  // ensure() loads vision+head once; failures are SURFACED on-screen (never a silent no-op)
  const ensure = async () => {
    if (_vis) return true;
    big("⏳ 모델 로딩 중…", "vision+head ONNX 다운로드/초기화", "#f0a73b");
    try { await load(PERF, status); big("✅ 준비완료 — 샘플 ×8 누르세요", null, "#36d399"); return true; }
    catch (e) { big("❌ 로딩 실패", (e && e.message) || String(e), "#f0a73b"); status(`로딩 실패: ${(e && e.message) || e}`); return false; }
  };

  async function runImages(getImg, count, label) {
    if (!(await ensure())) return;
    let done = 0;
    for (let i = 0; i < count; i++) {
      big(`⏳ 인코딩 중… ${i + 1}/${count}`, label, "#f0a73b");
      try { const img = await getImg(i); if (!img) continue; await encodeOne(img, PERF); done++; showStats(); }
      catch (e) { status(`인코딩 오류: ${e.message || e}`); }
    }
    if (!done) big('⚠ 인코딩된 이미지 없음', '"+ 사진"으로 갤러리 사진을 선택하세요', "#f0a73b");
    else { showStats(); status(`완료 · ${done}장`); }
  }

  box.querySelector("#pfImgPick").onclick = async () => { if (await ensure()) file.click(); };
  file.onchange = async () => {
    const files = [...file.files];
    await runImages((i) => fileToImage(files[i]), files.length, "갤러리 사진");
    file.value = "";
  };
  // sample = bundled same-origin JPEGs (/samples/0..N-1.jpg); no CORS taint, no file-picker
  box.querySelector("#pfImgSample").onclick = () =>
    runImages((i) => loadImageURL(`${SAMPLE_DIR}/${i}.jpg`).catch(() => null), SAMPLE_N, "번들 샘플");

  // EAGER PRELOAD while online (measurement is ONLINE — no airplane needed; latency is identical and
  // it avoids the SW-precache race). s0 fp32 (~46MB) loads fast; status + big readout show progress.
  if (navigator.onLine) { status("온라인 — 모델 로딩 시작…"); ensure(); }
  else { big("📴 오프라인", "온라인으로 1회 열어 모델 로드 후 측정", "#f0a73b"); }
}
