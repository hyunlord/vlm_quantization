"""Build Korean training pairs + a Korean test benchmark — CHEAP (no image re-embed).
coco_ko = COCO images (img emb already cached) + Korean captions. We embed only the
Korean captions (text) and pair with the cached COCO image embeddings by cocoid.

  train: coco_ko_train.jsonl (113K, leakage-safe train+restval) -> /tmp/coco_ko_pairs.pt
  test : coco_ko.jsonl Korean captions for the 5K COCO test images -> /tmp/coco_ko_test.pt
         (test Korean caps used ONLY for eval; training uses train split only -> no leak)
Stored L2-normalized (matches cc12m/oi pairs; train re-normalizes = no-op).
"""
import sys, os, re, json
import numpy as np, torch, torch.nn.functional as F
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
MODEL = "google/siglip2-so400m-patch14-384"
dev = "cuda" if torch.cuda.is_available() else "cpu"
def cocoid(p):
    m = re.search(r"_0*(\d+)\.jpg", p); return int(m.group(1)) if m else -1

# cached COCO image embeddings (train + test) keyed by cocoid
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
tr_img = F.normalize(TC["clean"].float(), dim=1); tr_ids = [int(x) for x in TC["ids"].tolist()]
id2row = {c: i for i, c in enumerate(tr_ids)}
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
print(f"COCO train imgs {len(tr_ids):,}, test imgs {len(te_ids):,}", flush=True)

from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
try: del bb.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
def embed(texts):
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 256):
            t = tok(texts[s:s+256], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            am = t.get("attention_mask")
            e = pool(bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)).float()
            out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out, 0)

# ---- train pairs ----
rows, kcaps = [], []
for e in (json.loads(l) for l in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl")):
    cid = cocoid(e["image_path"])
    if cid in id2row and e.get("captions"):
        rows.append(id2row[cid]); kcaps.append(e["captions"][0])
print(f"Korean train pairs matched: {len(rows):,}", flush=True)
kt = embed(kcaps)
torch.save({"img_emb": tr_img[torch.tensor(rows)], "txt_emb": kt}, "/tmp/coco_ko_pairs.pt")
print(f"saved /tmp/coco_ko_pairs.pt ({len(rows):,} pairs)", flush=True)

# ---- test Korean captions (for the 5K COCO test images, aligned to emb_cache order) ----
ko_full = {}
for e in (json.loads(l) for l in open(f"{REPO}/data/coco_ko/coco_ko.jsonl")):
    ko_full[cocoid(e["image_path"])] = e.get("captions", [])
test_kcaps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]
miss = sum(1 for c in test_kcaps if not c)
tt_ko = embed(test_kcaps)
torch.save({"txt_emb": tt_ko, "ids": EC["test"]["ids"]}, "/tmp/coco_ko_test.pt")
print(f"saved /tmp/coco_ko_test.pt ({len(test_kcaps):,} Korean test caps, missing={miss})", flush=True)
print("KOREAN_BUILD_DONE", flush=True)
