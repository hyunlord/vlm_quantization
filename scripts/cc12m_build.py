"""Post-recovery: verify alignment, build cc12m_index.npz, extract jpgs for serving.
Run AFTER recover_cc12m_paths.py finishes (/tmp/cc12m_paths.npy).

1. count check: len(paths) == len(big_emb)  (expect 806,560)
2. cosine gate: sample rows -> extract that image -> SigLIP2 vision emb -> cos with
   the stored embedding must be ~1.0 (proves path<->emb alignment). Abort if <0.95.
3. build /tmp/cc12m_index.npz {emb, paths('stem/key.jpg'), captions(llava_short)}
4. extract jpgs -> data/cc12m_imgs/{stem}/{key}.jpg  (grouped by tar, one open each)
"""
import sys, os, io, glob, gzip, json, tarfile
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
MODEL = "google/siglip2-so400m-patch14-384"
OUTDIR = f"{REPO}/data/cc12m_imgs"
paths = np.load("/tmp/cc12m_paths.npy", allow_pickle=True)
big = torch.load("/tmp/cc12m_pairs_big.pt", map_location="cpu")["img_emb"]   # (N,1152) normalized
print(f"paths={len(paths):,} emb={tuple(big.shape)}", flush=True)
assert len(paths) == big.shape[0], f"COUNT MISMATCH {len(paths)} != {big.shape[0]}"

def parts(p): d, stem, key = str(p).split("/"); return d, stem, key
def load_img(p):
    d, stem, key = parts(p)
    with tarfile.open(f"/tmp/{d}/{stem}.tar") as t:
        return Image.open(io.BytesIO(t.extractfile(t.getmember(f"{key}.jpg")).read())).convert("RGB")

# ---- cosine alignment gate ----
from transformers import AutoModel, SiglipImageProcessor
ip = SiglipImageProcessor.from_pretrained(MODEL)
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
try: del bb.text_model
except Exception: pass
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
rng = np.random.default_rng(0); idxs = sorted(rng.choice(len(paths), 8, replace=False).tolist())
coss = []
for i in idxs:
    px = ip(images=load_img(paths[i]), return_tensors="pt")["pixel_values"].cuda()
    with torch.no_grad():
        v = F.normalize(pool(bb.vision_model(pixel_values=px)).float(), dim=1)[0].cpu()
    c = float(v @ big[i]); coss.append(c); print(f"  row {i:,}: cos={c:.3f}", flush=True)
mean = float(np.mean(coss)); print(f"MEAN COSINE = {mean:.3f}", flush=True)
assert mean > 0.95, f"MISALIGNED (mean cos {mean:.3f}) — aborting before index/extract"
print("ALIGNMENT OK", flush=True)

# ---- captions (llava_short for display) ----
key2cap = {}
for cf in sorted(glob.glob("/tmp/cc12m_caps/**/*.jsonl.gz", recursive=True)) + sorted(glob.glob("/tmp/cc12m_caps/**/*.jsonl", recursive=True)):
    op = gzip.open if cf.endswith(".gz") else open
    with op(cf, "rt", encoding="utf-8") as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            k = d.get("key", d.get("__key__"))
            c = d.get("caption_llava_short") or d.get("caption") or d.get("caption_llava")
            if k is not None and c: key2cap[str(k)] = c
print(f"captions for display: {len(key2cap):,}", flush=True)

# ---- build index ----
ipaths, caps = [], []
for p in paths:
    _, stem, key = parts(p); ipaths.append(f"{stem}/{key}.jpg"); caps.append(key2cap.get(key, ""))
np.savez("/tmp/cc12m_index.npz", emb=big.numpy().astype(np.float32),
         paths=np.array(ipaths, dtype=object), captions=np.array(caps, dtype=object))
print(f"saved /tmp/cc12m_index.npz (caps {sum(1 for c in caps if c):,}/{len(caps):,})", flush=True)

# ---- extract jpgs (grouped by tar) ----
bytar = defaultdict(list)
for p in paths:
    d, stem, key = parts(p); bytar[(d, stem)].append(key)
done = 0
for (d, stem), keys in sorted(bytar.items()):
    od = f"{OUTDIR}/{stem}"; os.makedirs(od, exist_ok=True)
    want = {f"{k}.jpg": k for k in keys}
    with tarfile.open(f"/tmp/{d}/{stem}.tar") as t:
        for m in t:
            bn = os.path.basename(m.name)
            if bn in want:
                fo = f"{od}/{bn}"
                if not os.path.exists(fo):
                    with open(fo, "wb") as f: f.write(t.extractfile(m).read())
                done += 1
    print(f"  extracted {stem}: {done:,} total", flush=True)
print(f"EXTRACT_DONE {done:,} jpgs -> {OUTDIR}", flush=True)
