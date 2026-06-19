/* Stage A gate (cycle 6) — Node parity for the OFFLINE browser query path:
 *   transformers.js stock e5 (Xenova/multilingual-e5-small, q8/int8, mean-pool+L2)
 *   -> onnxruntime-node txt_h.onnx (fp32) -> continuous@1024 -> packBits (search.js, == pack_bits)
 *   -> hammingTopK (search.js) over the frozen 5K gallery codes.
 *
 * Confirms the ONNX/JS offline path reproduces PyTorch C1 (EN R@10 74.0 / KO 66.2):
 *   - R@10 (eval_korean 5K gallery) within ~1-2pt of C1,
 *   - per-query code closeness to PyTorch C1 codes (mean Hamming / % byte-identical),
 *   and reports e5 dtype, sizes, and Node encode latency. This is the exact pipeline app.js
 *   runs in the browser (same e5, same txt_h.onnx, same packBits/hammingTopK from search.js).
 *
 * Assets from web/export_txth_onnx.py (/tmp/parity): gallery.bin, queries.json, c1_en/ko.bin.
 * Setup + run (DGX): cd /tmp/v2spike_js && npm i @huggingface/transformers@4.2.0 onnxruntime-node
 *   cp <repo>/web/static/search.js ./search.mjs ; cp <repo>/web/static/onnx/txt_h.onnx /tmp/parity/
 *   DTYPE=q8 node <repo-or-copied>/hybrid_parity.mjs
 */
import { pipeline, env } from "@huggingface/transformers";
import * as ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { packBits, hammingTopK } from "./search.mjs";

const P = process.env.PARITY || "/tmp/parity";
const ONNX = process.env.ONNX || `${P}/txt_h.onnx`;
const DTYPE = process.env.DTYPE || "q8"; // q8=int8 e5 backbone; fp16/fp32 fallback
const CB = 128, BITS = 1024, K = 10;
env.allowLocalModels = false; // fetch stock e5 from the HF hub

function loadCodes(path) {
  const b = readFileSync(path);
  return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
}
function pairHamming(a, b, n) {
  const LUT = new Uint8Array(256);
  for (let i = 0; i < 256; i++) LUT[i] = (i & 1) + LUT[i >> 1];
  let tot = 0, ident = 0;
  for (let i = 0; i < n; i++) {
    let d = 0;
    for (let k = 0; k < CB; k++) d += LUT[a[i * CB + k] ^ b[i * CB + k]];
    tot += d; if (d === 0) ident++;
  }
  return { mean: tot / n, pctIdentical: (100 * ident / n).toFixed(1) };
}

const gallery = loadCodes(`${P}/gallery.bin`);
const meta = JSON.parse(readFileSync(`${P}/queries.json`, "utf-8"));
const ids = meta.ids, N = ids.length;
const c1 = { en: loadCodes(`${P}/c1_en.bin`), ko: loadCodes(`${P}/c1_ko.bin`) };
console.log(`[parity] gallery ${gallery.length / CB} rows | queries ${N} | e5 dtype=${DTYPE}`);

console.log("[parity] loading stock e5 (transformers.js) + txt_h.onnx (onnxruntime-node)...");
const extractor = await pipeline("feature-extraction", "Xenova/multilingual-e5-small", { dtype: DTYPE });
const sess = await ort.InferenceSession.create(ONNX);

async function encodeOffline(texts) {
  const out = new Uint8Array(texts.length * CB);
  const B = 64;
  let t0 = performance.now();
  for (let s = 0; s < texts.length; s += B) {
    const batch = texts.slice(s, s + B);
    const emb = await extractor(batch, { pooling: "mean", normalize: true }); // (b,384) L2
    const b = batch.length, D = emb.dims[1];
    const tensor = new ort.Tensor("float32", Float32Array.from(emb.data), [b, D]);
    const res = await sess.run({ emb: tensor });
    const cont = res.code.data; // Float32Array b*1024
    for (let i = 0; i < b; i++) {
      const code = packBits(cont.subarray(i * BITS, (i + 1) * BITS));
      out.set(code, (s + i) * CB);
    }
  }
  return { codes: out, msPerQuery: (performance.now() - t0) / texts.length };
}

const results = { dtype: DTYPE, c1_ref: { EN: 74.0, KO: 66.2 } };
for (const [LANG, key] of [["EN", "en"], ["KO", "ko"]]) {
  const { codes, msPerQuery } = await encodeOffline(meta[key]);
  let hit = 0;
  for (let i = 0; i < N; i++) {
    const q = codes.subarray(i * CB, (i + 1) * CB);
    const top = hammingTopK(gallery, N, CB, q, K);
    if (top.some((h) => h.idx === i)) hit++; // gallery row i == query i's gold image
  }
  const r10 = +(100 * hit / N).toFixed(2);
  const ph = pairHamming(codes, c1[key], N);
  results[LANG] = { R10: r10, R10_delta_vs_c1: +(r10 - results.c1_ref[LANG]).toFixed(2),
                    pair_hamming_vs_c1: +ph.mean.toFixed(2), pct_byte_identical_vs_c1: +ph.pctIdentical,
                    ms_per_query: +msPerQuery.toFixed(1) };
  console.log(`[parity] ${LANG}: R@10 ${r10} (Δc1 ${results[LANG].R10_delta_vs_c1 >= 0 ? "+" : ""}${results[LANG].R10_delta_vs_c1}) | ` +
    `code vs C1 mean Hamming ${results[LANG].pair_hamming_vs_c1}/1024 (${ph.pctIdentical}% identical) | ${results[LANG].ms_per_query} ms/q`);
}
const pass = Math.abs(results.EN.R10_delta_vs_c1) <= 2 && Math.abs(results.KO.R10_delta_vs_c1) <= 2;
console.log(`\n[parity] ${pass ? "STAGE A PASS" : "STAGE A CHECK"}: ONNX offline R@10 within ~2pt of C1? EN ${results.EN.R10_delta_vs_c1} / KO ${results.KO.R10_delta_vs_c1}`);
console.log("[parity] RESULT_JSON " + JSON.stringify(results));
