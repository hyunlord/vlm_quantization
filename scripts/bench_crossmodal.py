"""Cross-modal (text->image) confirmation: is binarization the culprit, or the hash head?

For each text query, rank the image corpus 3 ways and compare overlaps:
  A. emb cosine        : text_emb  · image_emb     (demo "float search" — the good one)
  B. float-hash        : text_logit· image_logit   (hash space, pre-sign float)
  C. symmetric bit     : sign(txt) · sign(img)      (current bit search)

Self-consistent hash space (everything from L2-normalized embeddings).
Key metric: overlap(C,B). If HIGH -> bits faithfully track float-hash -> binarization
is innocent; the divergence from emb (A) is the LEARNED hash head's doing.

Env: INDEX, HEADS, K(12), QUERIES(';'-sep)
"""
from __future__ import annotations
import os, sys
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

INDEX = os.environ.get("INDEX", "/tmp/oi_index_167k.npz")
HEADS = os.environ.get("HEADS", "/tmp/demo_hashheads.pt")
MODEL = "google/siglip2-so400m-patch14-384"
K = int(os.environ.get("K", "12"))
QUERIES = os.environ.get("QUERIES",
    "running lab;a labrador retriever dog running;a dog running in grass;library bookshelves").split(";")

dev = "cuda" if torch.cuda.is_available() else "cpu"
hh = torch.load(HEADS, map_location="cpu"); BITS = hh["bits"]; ki = BITS.index(1024)
img_h = NestedHashLayer(hh["embed"], hh["hidden"], BITS, 0.1); img_h.load_state_dict(hh["img_h"]); img_h.to(dev).eval()
txt_h = NestedHashLayer(hh["embed"], hh["hidden"], BITS, 0.1); txt_h.load_state_dict(hh["txt_h"]); txt_h.to(dev).eval()

idx = np.load(INDEX, allow_pickle=True)
img_emb = np.ascontiguousarray(idx["emb"].astype(np.float32))     # (N,1152) normalized
N = len(img_emb)

# image hash logits (1024) from normalized emb
print(f"N={N:,} | computing image hash logits ...", flush=True)
img_logit = np.empty((N, 1024), np.float32)
with torch.no_grad():
    for i in range(0, N, 8192):
        outs = img_h(torch.from_numpy(img_emb[i:i+8192]).to(dev))
        img_logit[i:i+8192] = outs[ki]["binary"].cpu().numpy()
img_bit = np.sign(img_logit).astype(np.float32)

# text encoder
from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
try: del bb.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

def topk(sc): return set(np.argpartition(-sc, K-1)[:K].tolist())
def ov(a, b): return len(a & b) / K * 100

print(f"\n{'query':38} | A∩B | A∩C | B∩C  (k={K})")
print("-"*70)
for q in QUERIES:
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e = pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                               attention_mask=t.get("attention_mask").to(dev) if t.get("attention_mask") is not None else None)).float()
        tl = txt_h(e.to(dev))[ki]["binary"][0].cpu().numpy()
    te = F.normalize(e, dim=1)[0].cpu().numpy()
    A = topk(img_emb @ te)                 # emb cosine
    B = topk(img_logit @ tl)               # float-hash
    C = topk(img_bit @ np.sign(tl))        # symmetric bit
    print(f"{q:38} | {ov(A,B):3.0f}%| {ov(A,C):3.0f}%| {ov(B,C):3.0f}%")
print("\nA=emb cosine(데모 float)  B=float-hash  C=bit  |  B∩C 높음 => 이진화 무죄")
