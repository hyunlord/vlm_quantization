/* Browser-side 1-bit search — hybrid query encoding (web/ v3).
 *
 * Search is unchanged: hammingTopK (search.js) over the flat index.bin, in the browser.
 * Two QUERY-ENCODING modes (toggle):
 *   - "정확/서버" (exact):   POST /encode_query -> so400m code (needs network). v1 path.
 *   - "오프라인" (offline):  in-browser  transformers.js stock e5 (q8) -> onnxruntime-web
 *                            txt_h.onnx (fp32) -> packBits -> 128-byte code. No backend.
 *
 * The offline path is byte-for-byte the pipeline validated in web/hybrid_parity.mjs
 * (Stage A: ONNX offline R@10 within ~2pt of PyTorch C1). packBits is shared from
 * search.js so the browser packs exactly like Python common.py:pack_bits.
 *
 * Offline mode is approximate (a DIFFERENT, independent encoder than the server) — good
 * recall, but results may differ from the exact/server mode. PWA (sw.js) caches the app +
 * index + model so offline mode works with no network.
 */
import { hammingTopK, packBits } from "./search.js";

const CDN_TFJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const CDN_ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.mjs";
const E5_MODEL = "Xenova/multilingual-e5-small";
const E5_DTYPE = "q8"; // int8 e5 (Stage A: passes parity). fp16/fp32 = larger fallback.

const State = {
  index: null, n: 0, codeBytes: 128, bits: 1024, meta: null, info: null,
  mode: "server", offline: null, offlineLoading: null,
};

const $ = (id) => document.getElementById(id);
const setStatus = (html) => { $("status").innerHTML = html; };

/* ---- index load (unchanged) -------------------------------------------- */
async function loadIndex() {
  try {
    State.info = await (await fetch("/data/index_info.json")).json();
    if (State.info.code_bytes) State.codeBytes = State.info.code_bytes;
    if (State.info.bits) State.bits = State.info.bits;
    if (State.info.head) $("headname").textContent = State.info.head;
  } catch (_) { /* optional */ }

  setStatus("메타데이터 로딩…");
  State.meta = await (await fetch("/data/meta.json")).json();

  setStatus("인덱스(index.bin) 로딩…");
  const resp = await fetch("/data/index.bin");
  const total = +resp.headers.get("Content-Length") || 0;
  const reader = resp.body.getReader();
  const chunks = []; let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    if (total) $("prog").style.width = (100 * received / total).toFixed(1) + "%";
  }
  const buf = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { buf.set(c, pos); pos += c.length; }
  State.index = buf;
  State.n = Math.floor(buf.length / State.codeBytes);
  $("prog").style.width = "100%";

  if (State.n !== State.meta.length) {
    setStatus(`<span class="warn">경고: 인덱스 행수 ${State.n} != meta ${State.meta.length}</span>`);
  } else {
    const mb = (buf.length / 1048576).toFixed(1);
    setStatus(`<span class="ok">준비 완료</span> · 코퍼스 <b>${State.n.toLocaleString()}</b>장 · index.bin ${mb} MB`);
  }
  $("go").disabled = false;
}

/* ---- exact/server encoding (v1, unchanged) ----------------------------- */
async function encodeServer(text) {
  const r = await fetch("/encode_query", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error("encode_query " + r.status);
  const j = await r.json();
  const raw = atob(j.code);
  const q = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) q[i] = raw.charCodeAt(i);
  return { q, encodeMs: j.encode_ms };
}

/* ---- offline encoding (in-browser, no backend) ------------------------- */
// Lazy-load the offline encoder once: transformers.js stock e5 (q8) + ort-web txt_h.onnx.
function loadOfflineEncoder() {
  if (State.offline) return Promise.resolve(State.offline);
  if (State.offlineLoading) return State.offlineLoading;
  State.offlineLoading = (async () => {
    setStatus("오프라인 인코더 로딩… (최초 1회, e5 모델 다운로드)");
    const [{ pipeline, env }, ort] = await Promise.all([import(CDN_TFJS), import(CDN_ORT)]);
    env.allowLocalModels = false;                 // fetch stock e5 from the HF hub (cached by SW)
    const extractor = await pipeline("feature-extraction", E5_MODEL, { dtype: E5_DTYPE });
    const session = await ort.InferenceSession.create("/onnx/txt_h.onnx");
    State.offline = { extractor, ort, session };
    setStatus(`<span class="ok">오프라인 인코더 준비 완료</span>`);
    return State.offline;
  })();
  return State.offlineLoading;
}

async function encodeOffline(text) {
  const { extractor, ort, session } = await loadOfflineEncoder();
  const t0 = performance.now();
  const emb = await extractor(text, { pooling: "mean", normalize: true });   // (1,384) L2
  const tensor = new ort.Tensor("float32", Float32Array.from(emb.data), [1, emb.dims[1]]);
  const cont = (await session.run({ emb: tensor })).code.data;               // Float32Array(1024)
  const q = packBits(cont);                                                  // 128 bytes, == pack_bits
  return { q, encodeMs: +(performance.now() - t0).toFixed(1) };
}

/* ---- search ------------------------------------------------------------ */
async function search(text) {
  if (!text.trim() || !State.index) return;
  $("go").disabled = true;
  const offline = State.mode === "offline";
  setStatus(offline ? "오프라인 인코딩 중…" : "쿼리 인코딩 중(서버)…");
  try {
    const { q, encodeMs } = offline ? await encodeOffline(text) : await encodeServer(text);
    if (q.length !== State.codeBytes) throw new Error(`code ${q.length}B != ${State.codeBytes}B`);
    const t0 = performance.now();
    const hits = hammingTopK(State.index, State.n, State.codeBytes, q, 30);
    const searchMs = performance.now() - t0;
    console.log(`[search:${State.mode}] encode ${encodeMs} ms, search ${searchMs.toFixed(1)} ms / ${State.n}`);
    $("stats").innerHTML =
      `<b>${offline ? "오프라인" : "서버"}</b> 인코딩 ${encodeMs} ms · 검색 <b>${searchMs.toFixed(1)} ms</b> · ` +
      `${State.n.toLocaleString()}장 · Top-${hits.length}` +
      (offline ? ` · <span class="warn">근사 모드</span>` : "");
    render(hits);
    setStatus(`<span class="ok">완료</span>`);
  } catch (e) {
    setStatus(`<span class="warn">오류: ${e.message}</span>`);
  } finally {
    $("go").disabled = false;
  }
}

function render(hits) {
  const grid = $("grid");
  grid.innerHTML = "";
  for (const h of hits) {
    const m = State.meta[h.idx];
    const sim = (1 - h.dist / State.bits) * 100;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML =
      `<div class="imgwrap"><img loading="lazy" src="${m.thumb}" alt=""></div>` +
      `<div class="info"><div class="cap">${escapeHtml(m.caption || m.id)}</div>` +
      `<div class="dist">d=${h.dist} · ${sim.toFixed(1)}%</div></div>`;
    grid.appendChild(card);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* ---- wire up ----------------------------------------------------------- */
const EXAMPLES = ["바닷가 강아지", "눈 덮인 산", "도시의 야경", "a plate of food", "two giraffes", "people riding bicycles"];
function initChips() {
  const box = $("chips");
  for (const ex of EXAMPLES) {
    const c = document.createElement("span");
    c.className = "chip"; c.textContent = ex;
    c.onclick = () => { $("q").value = ex; search(ex); };
    box.appendChild(c);
  }
}

function initModeToggle() {
  const sel = $("mode");
  if (!sel) return;
  sel.value = State.mode;
  sel.addEventListener("change", () => {
    State.mode = sel.value;
    if (State.mode === "offline") loadOfflineEncoder().catch((e) =>
      setStatus(`<span class="warn">오프라인 로딩 실패: ${e.message}</span>`));
  });
}

$("go").onclick = () => search($("q").value);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") search($("q").value); });
initChips();
initModeToggle();
loadIndex().catch((e) => setStatus(`<span class="warn">로딩 실패: ${e.message}</span>`));

// register the PWA service worker (offline support); harmless if unsupported
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch((e) => console.warn("SW reg failed", e));
}
