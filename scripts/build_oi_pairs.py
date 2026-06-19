"""Build OI (image_emb, text_emb) training pairs from Localized Narratives.

Image embeddings are ALREADY cached in oi_index (val+test, normalized SigLIP2
image-pooled). We only embed the narrative CAPTIONS with the SigLIP2 text tower
and pair them with the matching image embedding (by image_id). Cheap — no image
re-embedding.

Output /tmp/oi_pairs.pt: {"img_emb": (M,1152), "txt_emb": (M,1152)} (both L2-norm).

Env: OI_INDEX, NAR (comma-sep jsonl), OUT, MAXPER (max narratives per image, 0=all)
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
MODEL = "google/siglip2-so400m-patch14-384"
OI_INDEX = os.environ.get("OI_INDEX", "/tmp/oi_index_167k.npz")
NAR = os.environ.get("NAR", "/tmp/oi_nar_val.jsonl,/tmp/oi_nar_test.jsonl").split(",")
OUT = os.environ.get("OUT", "/tmp/oi_pairs.pt")
MAXPER = int(os.environ.get("MAXPER", "0"))
BATCH = int(os.environ.get("BATCH", "512"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

idx = np.load(OI_INDEX, allow_pickle=True)
emb = idx["emb"].astype(np.float32)                      # normalized image emb
id2row = {}
for i, p in enumerate(idx["paths"]):
    iid = os.path.splitext(os.path.basename(str(p)))[0]
    id2row[iid] = i
print(f"index images: {len(emb)} | embedding captions on {dev}", flush=True)

rows, caps = [], []
per = {}
for nf in NAR:
    if not os.path.exists(nf):
        print(f"  WARN missing {nf}"); continue
    for line in open(nf):
        try:
            d = json.loads(line)
        except Exception:
            continue
        iid = d.get("image_id"); cap = d.get("caption", "").strip()
        if not cap or iid not in id2row:
            continue
        if MAXPER:
            c = per.get(iid, 0)
            if c >= MAXPER:
                continue
            per[iid] = c + 1
        rows.append(id2row[iid]); caps.append(cap)
print(f"matched {len(rows)} (image,caption) pairs", flush=True)

from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
try: del bb.vision_model; torch.cuda.empty_cache()
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

txt = np.empty((len(caps), emb.shape[1]), np.float32)
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(0, len(caps), BATCH):
        t = tok(caps[i:i+BATCH], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        e = pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                               attention_mask=am.to(dev) if am is not None else None)).float()
        txt[i:i+BATCH] = F.normalize(e, dim=1).cpu().numpy()
        if i % (BATCH * 20) == 0:
            r = (i + BATCH) / (time.perf_counter() - t0)
            print(f"  {i+BATCH}/{len(caps)} ({r:.0f} cap/s)", flush=True)

img = emb[np.array(rows)]                                  # already normalized
torch.save({"img_emb": torch.from_numpy(img), "txt_emb": torch.from_numpy(txt)}, OUT)
print(f"PAIRS_DONE: {len(rows)} pairs -> {OUT} ({len(caps)/(time.perf_counter()-t0):.0f} cap/s)", flush=True)
