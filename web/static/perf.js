/* Offline-path latency panel — PURE INSTRUMENTATION. Loaded ONLY when the URL has ?perf=1
 * (app.js dynamic-imports it), so normal UX never touches this file and search results are
 * unaffected (it only reads performance.now() timings and renders a separate overlay).
 *
 * Stages measured by app.js (boundaries match the paper):
 *   encode = tokenize + e5-small int8 (ORT/transformers.js)   [extractor(text)]
 *   head   = txt_h' fp32 ONNX (tanh) + packBits -> 1024-bit code
 *   search = 50K-index Hamming top-10 (hammingTopK)
 * Per query: {encode, head, search, total, qlen, mode, online, ts}. Panel shows last query +
 * running MEDIAN/p90/n (median emphasized — mobile latency has a heavy tail).
 */

// ---- pure stat helpers (exported for the Node unit test; no DOM) ----
export function quantile(values, p) {
  const a = values.filter((v) => typeof v === "number" && isFinite(v)).slice().sort((x, y) => x - y);
  if (!a.length) return null;
  if (a.length === 1) return a[0];
  const idx = p * (a.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? a[lo] : a[lo] + (a[hi] - a[lo]) * (idx - lo);
}
export const median = (v) => quantile(v, 0.5);
export const p90 = (v) => quantile(v, 0.9);
export function statsFor(records, key) {
  const xs = records.map((r) => r[key]).filter((v) => typeof v === "number" && isFinite(v));
  return { n: xs.length, median: round1(median(xs)), p90: round1(p90(xs)) };
}
function round1(v) { return v == null ? null : Math.round(v * 10) / 10; }

class Perf {
  constructor() {
    this.records = [];
    this.env = {};                 // ep, simd, threads + filled from navigator on mount
    this.cold = {};                // {index_ms, encoder_ms, sw_controlled}
    this.el = null;
    // ---- image-indexing path (perf_img.js, ?img=1) — measures cost to hash a NEW photo:
    // vis = MobileCLIP2-S2 vision ONNX; imgh = img_h' fp32 ONNX + packBits. No search (indexing). ----
    this.imgRecords = [];          // {vis, imgh, total, ep, ts, online}
    this.imgEnv = {};              // {ep, dim, onnx_mb, input}
    this.imgCold = {};             // {load_ms}
  }

  setEnv(e) { Object.assign(this.env, e); this.render(); }
  setCold(c) { Object.assign(this.cold, c); this.render(); }
  setImgEnv(e) { Object.assign(this.imgEnv, e); this.render(); }
  setImgCold(c) { Object.assign(this.imgCold, c); this.render(); }

  record(r) {
    r.ts = Date.now();
    r.online = navigator.onLine;
    this.records.push(r);
    this.render();
  }

  recordImg(r) {
    r.ts = Date.now();
    r.online = navigator.onLine;
    this.imgRecords.push(r);
    this.render();
  }

  reset() { this.records = []; this.imgRecords = []; this.render(); }

  envSnapshot() {
    return {
      ...this.env,
      userAgent: navigator.userAgent,
      onLine: navigator.onLine,
      hardwareConcurrency: navigator.hardwareConcurrency || null,
    };
  }

  snapshot() {
    return {
      env: this.envSnapshot(),
      coldload: this.cold,
      n: this.records.length,
      stats: {
        encode: statsFor(this.records, "encode"),
        head: statsFor(this.records, "head"),
        search: statsFor(this.records, "search"),
        total: statsFor(this.records, "total"),
      },
      records: this.records,
      image: {
        env: { ...this.imgEnv, onLine: navigator.onLine },
        coldload: this.imgCold,
        n: this.imgRecords.length,
        stats: {
          vis: statsFor(this.imgRecords, "vis"),
          imgh: statsFor(this.imgRecords, "imgh"),
          total: statsFor(this.imgRecords, "total"),
        },
        records: this.imgRecords,
      },
    };
  }

  // ---- overlay (built once; only when ?perf=1) ----
  mount() {
    if (this.el) return;
    const wrap = document.createElement("div");
    wrap.id = "perfPanel";
    wrap.setAttribute("style", [
      "position:fixed", "right:8px", "bottom:8px", "z-index:99999",
      "width:248px", "max-width:46vw", "background:rgba(8,10,16,.93)",
      "color:#e7ebf3", "border:1px solid #2a3247", "border-radius:10px",
      "font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace", "padding:8px 9px",
      "box-shadow:0 4px 16px rgba(0,0,0,.4)",
    ].join(";"));
    wrap.innerHTML =
      '<div style="font-weight:700;color:#5b9dff;margin-bottom:4px">offline perf · <span id="pfN">0</span> q</div>' +
      '<div id="pfLast" style="color:#8b94a7">no query yet</div>' +
      '<div id="pfMed" style="margin-top:4px"></div>' +
      '<div id="pfEnv" style="margin-top:5px;color:#8b94a7;font-size:10px;word-break:break-word"></div>' +
      '<div id="pfImgSec" style="display:none;margin-top:8px;padding-top:6px;border-top:1px solid #2a3247">' +
      '<div style="font-weight:700;color:#36d399;margin-bottom:3px">on-device image index · <span id="pfImgN">0</span> img</div>' +
      '<div id="pfImgLast" style="color:#8b94a7">no image yet</div>' +
      '<div id="pfImgMed" style="margin-top:3px"></div>' +
      '<div id="pfImgEnv" style="margin-top:4px;color:#8b94a7;font-size:10px;word-break:break-word"></div>' +
      '<div id="pfImgUI" style="margin-top:5px"></div>' +
      "</div>" +
      '<div style="margin-top:6px;display:flex;gap:6px">' +
      '<button id="pfReset" style="flex:1;font:inherit;padding:4px;border:1px solid #2a3247;border-radius:6px;background:#141821;color:#e7ebf3;cursor:pointer">Reset</button>' +
      '<button id="pfCopy" style="flex:1;font:inherit;padding:4px;border:1px solid #2a3247;border-radius:6px;background:#141821;color:#e7ebf3;cursor:pointer">Copy</button>' +
      "</div>" +
      '<textarea id="pfOut" readonly style="display:none;width:100%;height:96px;margin-top:6px;font:10px ui-monospace,monospace;background:#0b0d12;color:#9fb0cc;border:1px solid #2a3247;border-radius:6px"></textarea>';
    document.body.appendChild(wrap);
    this.el = wrap;
    wrap.querySelector("#pfReset").onclick = () => this.reset();
    wrap.querySelector("#pfCopy").onclick = () => this.copy();
    this.render();
  }

  copy() {
    const json = JSON.stringify(this.snapshot(), null, 2);
    const ta = this.el.querySelector("#pfOut");
    ta.style.display = "block";
    ta.value = json;
    ta.focus(); ta.select();
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(json).catch(() => {});
    }
  }

  render() {
    if (!this.el) return;
    const q = (k) => this.el.querySelector(k);
    q("#pfN").textContent = String(this.records.length);
    const last = this.records[this.records.length - 1];
    q("#pfLast").innerHTML = last
      ? `last: enc <b>${fmt(last.encode)}</b> · head <b>${fmt(last.head)}</b> · search <b>${fmt(last.search)}</b> · <b style="color:#36d399">tot ${fmt(last.total)}</b> ms`
      : "no query yet";
    const s = this.snapshot().stats;
    q("#pfMed").innerHTML =
      `median enc ${fmt(s.encode.median)} · head ${fmt(s.head.median)} · search ${fmt(s.search.median)} · <b>tot ${fmt(s.total.median)}</b><br>` +
      `<span style="color:#8b94a7">p90 enc ${fmt(s.encode.p90)} · head ${fmt(s.head.p90)} · search ${fmt(s.search.p90)} · tot ${fmt(s.total.p90)}</span>`;
    const e = this.envSnapshot();
    const cold = this.cold.index_ms != null || this.cold.encoder_ms != null
      ? ` · cold idx ${fmt(this.cold.index_ms)}/enc ${fmt(this.cold.encoder_ms)} ms${this.cold.sw_controlled ? " (SW)" : ""}` : "";
    q("#pfEnv").innerHTML =
      `EP <b style="color:${e.onLine ? "#f0a73b" : "#36d399"}">${e.ep || "?"}</b>` +
      `${e.simd != null ? ` simd:${e.simd ? 1 : 0}` : ""}${e.threads != null ? ` thr:${e.threads}` : ""}` +
      ` · hc:${e.hardwareConcurrency ?? "?"} · online:<b style="color:${e.onLine ? "#f0a73b" : "#36d399"}">${e.onLine}</b>` +
      ` · ${e.bits || "?"}b/${(e.n || 0).toLocaleString()}${cold}`;

    // ---- image-indexing section (revealed by perf_img.js when ?img=1) ----
    const sec = q("#pfImgSec");
    if (sec && (this.imgRecords.length || Object.keys(this.imgEnv).length)) {
      sec.style.display = "block";
      q("#pfImgN").textContent = String(this.imgRecords.length);
      const li = this.imgRecords[this.imgRecords.length - 1];
      q("#pfImgLast").innerHTML = li
        ? `last: vis <b>${fmt(li.vis)}</b> · head <b>${fmt(li.imgh)}</b> · <b style="color:#36d399">tot ${fmt(li.total)}</b> ms`
        : "no image yet";
      const is = this.snapshot().image.stats;
      q("#pfImgMed").innerHTML =
        `median vis ${fmt(is.vis.median)} · head ${fmt(is.imgh.median)} · <b>tot ${fmt(is.total.median)}</b><br>` +
        `<span style="color:#8b94a7">p90 vis ${fmt(is.vis.p90)} · head ${fmt(is.imgh.p90)} · tot ${fmt(is.total.p90)}</span>`;
      const ie = this.imgEnv;
      const icold = this.imgCold.load_ms != null ? ` · cold ${fmt(this.imgCold.load_ms)} ms` : "";
      q("#pfImgEnv").innerHTML =
        `EP <b style="color:#5b9dff">${ie.ep || "?"}</b> · ${ie.model || "vis"} ${ie.input || "?"}px` +
        ` · ${ie.dim || "?"}d · onnx ${ie.onnx_mb ?? "?"}MB${icold}`;
    }
  }

  // perf_img.js calls this to reveal the image section and get its UI container.
  imgContainer() {
    if (!this.el) this.mount();
    const sec = this.el.querySelector("#pfImgSec");
    if (sec) sec.style.display = "block";
    return this.el.querySelector("#pfImgUI");
  }
}

function fmt(v) { return v == null ? "–" : (Math.round(v * 10) / 10).toFixed(1); }

export const PERF = new Perf();
