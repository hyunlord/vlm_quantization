/* v2 feasibility spike (cycle 3) — tokenizer-in-JS axis.
 *
 * Confirms the SigLIP2 (Gemma) tokenizer runs in a JS runtime via transformers.js and
 * produces token ids IDENTICAL to Python's web/common.py tokenizer. Reads the Python
 * reference written by web/spike_probe.py (/tmp/tok_ref.json) and compares.
 *
 * The HF repo google/siglip2-so400m-patch14-384 ships tokenizer.json (fast), which
 * transformers.js AutoTokenizer loads directly — no ONNX/model weights needed for this.
 *
 * Setup + run (on DGX or any Node host with network to the HF hub):
 *   mkdir -p /tmp/v2spike_js && cd /tmp/v2spike_js && npm init -y
 *   npm i @huggingface/transformers@4.2.0
 *   node /path/to/web/spike_tok_check.mjs           # needs /tmp/tok_ref.json from spike_probe.py
 *
 * Result (recorded in web/V2_SPIKE.md): ALL match for KO+EN samples (incl EOS=1, no BOS).
 */
import { AutoTokenizer } from "@huggingface/transformers";
import { readFileSync } from "node:fs";

const REF = process.env.TOK_REF || "/tmp/tok_ref.json";
const ref = JSON.parse(readFileSync(REF, "utf-8"));
const tok = await AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384");

let allMatch = true;
for (const [s, refIds] of Object.entries(ref)) {
  const refNon = refIds.filter((x) => x !== 0); // strip pad(0); keep real tokens incl EOS(1)
  const jsNon = tok.encode(s).filter((x) => x !== 0); // transformers.js -> number[]
  const match = JSON.stringify(refNon) === JSON.stringify(jsNon);
  allMatch = allMatch && match;
  console.log(`match=${match}  ${JSON.stringify(s)}`);
  console.log(`   py: ${JSON.stringify(refNon)}`);
  console.log(`   js: ${JSON.stringify(jsNon)}`);
}
console.log(allMatch ? "\nTOKENIZER_JS_MATCH: ALL" : "\nTOKENIZER_JS_MATCH: DIFFERS (see above)");
process.exit(allMatch ? 0 : 1);
