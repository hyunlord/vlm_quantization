/* Browser-side 1-bit search (web/ v1).
 *
 * Loads the flat packed index (index.bin) + row-aligned meta.json once, then for each
 * query asks the server only to ENCODE the text into a 1024-bit code (POST /encode_query)
 * and runs Hamming Top-K search locally. The server never searches.
 *
 * The search (hammingTopK) is a pure function kept free of DOM/network so it can be
 * moved into a Web Worker later without changes. It mirrors web/verify_parity.py
 * byte-for-byte: 256-entry uint8 popcount LUT, dist = sum(LUT[a^b]), and ties are
 * broken by ascending index — identical to the faiss-parity reference.
 */
"use strict";

// 256-entry popcount lookup table (popcount of each byte value).
const LUT = (() => {
  const t = new Uint8Array(256);
  for (let i = 0; i < 256; i++) t[i] = (i & 1) + t[i >> 1];
  return t;
})();

const State = { index: null, n: 0, codeBytes: 128, meta: null, info: null };

/* ---- pure search core (Worker-ready) ----------------------------------- */
// index: Uint8Array of n*cb bytes; q: Uint8Array(cb). Returns [{idx,dist}] sorted
// by (dist asc, idx asc), length <= K. Uses a bounded max-heap of capacity K.
function hammingTopK(index, n, cb, q, K) {
  const hd = new Uint16Array(K);   // heap distances (root = current worst kept)
  const hi = new Int32Array(K);    // heap indices
  let hs = 0;                      // heap size
  // a "worse" than b iff larger dist, or equal dist & larger index
  const worse = (aD, aI, bD, bI) => aD > bD || (aD === bD && aI > bI);

  for (let i = 0; i < n; i++) {
    const off = i * cb;
    let d = 0;
    for (let k = 0; k < cb; k++) d += LUT[index[off + k] ^ q[k]];

    if (hs < K) {                                  // grow heap
      let c = hs++;
      hd[c] = d; hi[c] = i;
      while (c > 0) {                              // sift up (max-heap)
        const p = (c - 1) >> 1;
        if (worse(hd[c], hi[c], hd[p], hi[p])) {
          const td = hd[c]; hd[c] = hd[p]; hd[p] = td;
          const ti = hi[c]; hi[c] = hi[p]; hi[p] = ti;
          c = p;
        } else break;
      }
    } else if (d < hd[0] || (d === hd[0] && i < hi[0])) {  // better than worst -> replace root
      hd[0] = d; hi[0] = i;
      let c = 0;                                   // sift down
      for (;;) {
        const l = 2 * c + 1, r = l + 1; let m = c;
        if (l < hs && worse(hd[l], hi[l], hd[m], hi[m])) m = l;
        if (r < hs && worse(hd[r], hi[r], hd[m], hi[m])) m = r;
        if (m === c) break;
        const td = hd[c]; hd[c] = hd[m]; hd[m] = td;
        const ti = hi[c]; hi[c] = hi[m]; hi[m] = ti;
        c = m;
      }
    }
  }
  const out = [];
  for (let i = 0; i < hs; i++) out.push({ idx: hi[i], dist: hd[i] });
  out.sort((a, b) => a.dist - b.dist || a.idx - b.idx);
  return out;
}

/* ---- loading ------------------------------------------------------------ */
const $ = (id) => document.getElementById(id);
const setStatus = (html) => { $("status").innerHTML = html; };

async function loadIndex() {
  try {
    State.info = await (await fetch("/data/index_info.json")).json();
    if (State.info.code_bytes) State.codeBytes = State.info.code_bytes;
    if (State.info.head) $("headname").textContent = State.info.head;
  } catch (_) { /* index_info is optional */ }

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
    setStatus(`<span class="warn">경고: 인덱스 행수 ${State.n} != meta ${State.meta.length} (정렬 깨짐)</span>`);
  } else {
    const mb = (buf.length / 1048576).toFixed(1);
    setStatus(`<span class="ok">준비 완료</span> · 코퍼스 <b>${State.n.toLocaleString()}</b>장 · ` +
              `index.bin ${mb} MB · ${State.codeBytes} B/이미지`);
  }
  $("go").disabled = false;
}

/* ---- query -------------------------------------------------------------- */
async function encodeQuery(text) {
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

async function search(text) {
  if (!text.trim() || !State.index) return;
  $("go").disabled = true;
  setStatus("쿼리 인코딩 중…");
  try {
    const { q, encodeMs } = await encodeQuery(text);
    const t0 = performance.now();
    const hits = hammingTopK(State.index, State.n, State.codeBytes, q, 30);
    const searchMs = performance.now() - t0;
    console.log(`[search] ${searchMs.toFixed(1)} ms over ${State.n} images (encode ${encodeMs} ms)`);
    $("stats").innerHTML =
      `검색 <b>${searchMs.toFixed(1)} ms</b> · ${State.n.toLocaleString()}장 · ` +
      `서버 인코딩 ${encodeMs} ms · Top-${hits.length}`;
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
  const bits = (State.info && State.info.bits) || State.codeBytes * 8;
  for (const h of hits) {
    const m = State.meta[h.idx];
    const sim = (1 - h.dist / bits) * 100;
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

/* ---- wire up ------------------------------------------------------------ */
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

$("go").onclick = () => search($("q").value);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") search($("q").value); });
initChips();
loadIndex().catch((e) => setStatus(`<span class="warn">로딩 실패: ${e.message}</span>`));
