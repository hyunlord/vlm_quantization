/* Node parity test (Acceptance Criteria §6.4) using the ACTUAL shipped JS.
 *
 * Loads web/static/search.js — the exact module the browser runs — and exercises its
 * hammingTopK over the real index.bin, then compares to faiss IndexBinaryFlat ground
 * truth produced by verify_parity.py. This closes the gap that verify_parity.py only
 * proves a *Python re-implementation* matches faiss; here the real JavaScript does.
 *
 * Usage (after `verify_parity.py --dump-js-queries /tmp/js_queries.json`):
 *   cd ~/github/vlm_quantization
 *   node web/verify_js.mjs --static web/static --queries /tmp/js_queries.json [--k 10]
 *
 * js_queries.json: [{ text, code_b64, faiss_pairs: [[dist,id], ...] }]  (faiss top-k)
 */
import { readFileSync } from "node:fs";
import { hammingTopK } from "./static/search.js";

const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
};
const staticDir = opt("--static", "web/static");
const queriesPath = opt("--queries", "/tmp/js_queries.json");
const K = parseInt(opt("--k", "10"), 10);
const CB = 128; // 1024-bit -> 128 bytes/row

// Load the flat index exactly as the browser does (raw bytes).
const buf = readFileSync(`${staticDir}/data/index.bin`);
const index = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
const n = Math.floor(index.length / CB);
const queries = JSON.parse(readFileSync(queriesPath, "utf-8"));
console.log(`[verify-js] index.bin: ${n.toLocaleString()} rows x ${CB} B | queries: ${queries.length}`);

let matched = 0;
const times = [];
const mismatches = [];
for (const q of queries) {
  const raw = Buffer.from(q.code_b64, "base64");
  const code = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);

  const t0 = performance.now();
  const hits = hammingTopK(index, n, CB, code, K);
  times.push(performance.now() - t0);

  // tie-immune compare: sort both as (dist, id)
  const jsPairs = hits.map((h) => [h.dist, h.idx]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const fxPairs = q.faiss_pairs.map((p) => [p[0], p[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const same = jsPairs.length === fxPairs.length &&
    jsPairs.every((p, i) => p[0] === fxPairs[i][0] && p[1] === fxPairs[i][1]);
  if (same) matched++;
  else mismatches.push({ text: q.text, js: jsPairs, faiss: fxPairs });
}

const mean = times.reduce((a, b) => a + b, 0) / times.length;
const max = Math.max(...times);
console.log(`[verify-js] §6.4 REAL app.js search.js vs faiss IndexBinaryFlat top-${K}: ` +
  `${matched}/${queries.length} exact match`);
console.log(`[verify-js] §6.3 actual-JS search time over ${n.toLocaleString()} rows (Node): ` +
  `mean ${mean.toFixed(1)} ms, max ${max.toFixed(1)} ms`);
for (const m of mismatches.slice(0, 5)) {
  console.log(`  MISMATCH ${JSON.stringify(m.text)}\n    js   : ${JSON.stringify(m.js)}\n    faiss: ${JSON.stringify(m.faiss)}`);
}
const ok = matched === queries.length;
console.log(ok ? "\n[verify-js] JS PARITY OK" : "\n[verify-js] JS PARITY FAILED");
process.exit(ok ? 0 : 1);
