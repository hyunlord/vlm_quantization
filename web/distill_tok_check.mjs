/* v2 distill spike (cycle 4) — student tokenizer-in-JS axis.
 *
 * Confirms the distilled student's tokenizer (intfloat/multilingual-e5-small = XLM-R
 * SentencePiece) runs in a JS runtime via transformers.js and matches Python token ids
 * for KO+EN samples. Reads the Python reference at /tmp/e5_tok_ref.json.
 *
 * Generate the ref (on DGX):
 *   .venv/bin/python -c "import json;from transformers import AutoTokenizer; \
 *     t=AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small'); \
 *     json.dump({s:t(s)['input_ids'] for s in ['바닷가 강아지','a dog on the beach','two giraffes']}, \
 *     open('/tmp/e5_tok_ref.json','w'), ensure_ascii=False)"
 * Setup + run:
 *   cd /tmp/v2spike_js && npm i @huggingface/transformers@4.2.0
 *   node /path/to/web/distill_tok_check.mjs
 */
import { AutoTokenizer } from "@huggingface/transformers";
import { readFileSync } from "node:fs";

const REF = process.env.TOK_REF || "/tmp/e5_tok_ref.json";
const ref = JSON.parse(readFileSync(REF, "utf-8"));
const tok = await AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small");

let allMatch = true;
for (const [s, refIds] of Object.entries(ref)) {
  const jsIds = tok.encode(s); // number[]
  const match = JSON.stringify(refIds) === JSON.stringify(jsIds);
  allMatch = allMatch && match;
  console.log(`match=${match}  ${JSON.stringify(s)}`);
  console.log(`   py: ${JSON.stringify(refIds)}`);
  console.log(`   js: ${JSON.stringify(jsIds)}`);
}
console.log(allMatch ? "\nE5_TOKENIZER_JS_MATCH: ALL" : "\nE5_TOKENIZER_JS_MATCH: DIFFERS");
process.exit(allMatch ? 0 : 1);
