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
const HEAD_URL = "/onnx/img_h_mobileclip2-s2.onnx";
const META_URL = "/onnx/vis_mobileclip2-s2.meta.json";
// Vision ONNX variants. fp32 is the FAITHFUL path (matches the trained code; runs on WebGPU+wasm).
// int8 is ~37MB but dynamic-quant drifts ~35% of the 1024 bits on this conv-heavy FastViT arch —
// LATENCY-ONLY, not a faithful index (opt in with ?imgvis=int8). fp16 auto-convert hits a FastViT
// Cast-node bug (omitted). Default = fp32 for correct on-device codes.
const VIS_FP32 = "/onnx/vis_mobileclip2-s2.onnx";
const VIS_INT8 = "/onnx/vis_mobileclip2-s2.int8.onnx";

let _ort = null, _vis = null, _head = null, _meta = null, _ep = null;

async function pickEP(ort) {
  const want = new URLSearchParams(location.search).get("imgep");
  if (want === "wasm") return "wasm";
  if (want === "webgpu") return "webgpu";
  // auto: try WebGPU if the browser exposes it, else wasm
  return (typeof navigator !== "undefined" && navigator.gpu) ? "webgpu" : "wasm";
}

async function head(url) {  // does a vision file exist? (HEAD; fall back to fp32 if not)
  try { return (await fetch(url, { method: "HEAD" })).ok; } catch (_) { return false; }
}

async function load(PERF, setStatus) {
  if (_vis) return;
  const t0 = performance.now();
  setStatus("이미지 인코더 로딩… (vision + head ONNX, 최초 1회)");
  _ort = await import(CDN_ORT);
  _meta = await (await fetch(META_URL)).json().catch(() => ({ input: 256, resize: 256, crop: 256, dim: 512 }));
  _ep = await pickEP(_ort);

  const wantInt8 = new URLSearchParams(location.search).get("imgvis") === "int8";
  let visUrl = (wantInt8 && (await head(VIS_INT8))) ? VIS_INT8 : VIS_FP32;
  const providers = _ep === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  _vis = await _ort.InferenceSession.create(visUrl, { executionProviders: providers });
  _head = await _ort.InferenceSession.create(HEAD_URL, { executionProviders: ["wasm"] });

  // warmup (WebGPU shader compile / wasm JIT) — not recorded as a real measurement
  const warm = new _ort.Tensor("float32", new Float32Array(3 * _meta.input * _meta.input), [1, 3, _meta.input, _meta.input]);
  const we = (await _vis.run({ px: warm }))[_vis.outputNames[0]];
  await _head.run({ emb: we });

  const onnxMb = await fetch(visUrl).then((r) => +(r.headers.get("Content-Length") || 0) / 1e6).catch(() => null);
  PERF.setImgEnv({ ep: _ep, model: "MobileCLIP2-S2", input: _meta.input, dim: _meta.dim,
    onnx_mb: onnxMb ? Math.round(onnxMb) : null, vis_file: visUrl.split("/").pop(),
    faithful: visUrl === VIS_FP32 });
  PERF.setImgCold({ load_ms: +(performance.now() - t0).toFixed(1),
    sw_controlled: !!(navigator.serviceWorker && navigator.serviceWorker.controller) });
  setStatus(`<span style="color:#36d399">이미지 인코더 준비 완료</span> · EP ${_ep}`);
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

export async function initImgPanel(PERF) {
  const box = PERF.imgContainer();
  if (!box) return;
  box.innerHTML =
    '<input id="pfImgFile" type="file" accept="image/*" multiple style="display:none">' +
    '<div style="display:flex;gap:6px">' +
    '<button id="pfImgPick" style="flex:1;font:inherit;padding:4px;border:1px solid #2a3247;border-radius:6px;background:#141821;color:#e7ebf3;cursor:pointer">+ 사진 인코딩</button>' +
    '<button id="pfImgSample" style="flex:1;font:inherit;padding:4px;border:1px solid #2a3247;border-radius:6px;background:#141821;color:#e7ebf3;cursor:pointer">샘플 ×8</button>' +
    "</div><div id='pfImgStatus' style='margin-top:4px;color:#8b94a7;font-size:10px'></div>";
  const status = (h) => { const s = box.querySelector("#pfImgStatus"); if (s) s.innerHTML = h; };
  const file = box.querySelector("#pfImgFile");

  const ensure = () => load(PERF, status);
  box.querySelector("#pfImgPick").onclick = async () => { await ensure(); file.click(); };
  file.onchange = async () => {
    await ensure();
    const files = [...file.files];
    for (let i = 0; i < files.length; i++) {
      status(`인코딩 ${i + 1}/${files.length}…`);
      try { const img = await fileToImage(files[i]); await encodeOne(img, PERF); }
      catch (e) { status(`오류: ${e.message || e}`); }
    }
    status(`완료 · ${files.length}장`);
    file.value = "";
  };

  // "sample" = re-encode the first gallery thumbnail K times (needs network for the thumb; for a
  // pure-offline phone run use "+ 사진 인코딩" with local photos). Useful as a desktop smoke path.
  box.querySelector("#pfImgSample").onclick = async () => {
    await ensure();
    let thumb = null;
    try { const meta = await (await fetch("/data/meta.json")).json(); thumb = meta[0] && meta[0].thumb; } catch (_) {}
    if (!thumb) { status("샘플 썸네일 없음 — 사진을 직접 선택하세요"); return; }
    for (let i = 0; i < 8; i++) {
      status(`샘플 인코딩 ${i + 1}/8…`);
      try {
        const img = await new Promise((res, rej) => { const im = new Image(); im.crossOrigin = "anonymous";
          im.onload = () => res(im); im.onerror = rej; im.src = thumb + (thumb.includes("?") ? "&" : "?") + "r=" + i; });
        await encodeOne(img, PERF);
      } catch (e) { status(`샘플 오류(원격 썸네일 CORS?): ${e.message || e}`); break; }
    }
    status("샘플 완료");
  };
}
