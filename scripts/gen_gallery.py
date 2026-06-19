"""GPU-free retrieval galleries for the experiment HTML.

For a few themed SEED images (picked via val narratives keywords), retrieve top-k
similar images by 3 methods over the OI 167K corpus:
  float  : SigLIP2 emb cosine (gold)
  bit-base   : baseline hash head (coco-only)
  bit-CC12M  : CC12M-trained hash head
All CPU (cached emb + tiny head forward) — no GPU, no contention with the embed.
Outputs /tmp/gallery.json + prints the image list to copy.
"""
from __future__ import annotations
import os, sys, json, glob, gzip
import numpy as np, torch

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

idx = np.load("/tmp/oi_index_167k.npz", allow_pickle=True)
emb = np.ascontiguousarray(idx["emb"].astype(np.float32))
paths = [str(p) for p in idx["paths"]]
N = len(emb)
id2row = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)}
embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

def load_head(p):
    hh = torch.load(p, map_location="cpu")
    h = NestedHashLayer(hh["embed"], hh["hidden"], hh["bits"], 0.1)
    h.load_state_dict(hh["img_h"]); h.eval()
    bi = hh["bits"].index(256) if 256 in hh["bits"] else len(hh["bits"]) - 1
    return h, bi

hb, bi = load_head("/tmp/broaden_head_coco.pt")
hc, _ = load_head("/tmp/broaden_head_coco_oi_rkd_crovca.pt")

def codes(h):
    out = []
    with torch.no_grad():
        for i in range(0, N, 16384):
            out.append(h(torch.from_numpy(emb[i:i+16384]))[bi]["binary"].numpy().astype(np.float32))
    return np.concatenate(out, 0)
print("computing hash codes (cpu)...", flush=True)
cb, cc = codes(hb), codes(hc)

# themed seeds via val narratives keywords (diverse, many)
KW = {"강아지": ["dog running", "a dog is", "dog standing"],
      "고양이": ["cat is", "a cat ", "kitten"],
      "도서관/책장": ["bookshelf", "library", "on the shelves"],
      "케이크/디저트": ["cake", "birthday", "dessert"],
      "피자/음식": ["pizza", "plate of food", "a burger"],
      "우산": ["umbrella"],
      "말 탄 사람": ["riding a horse", "on a horse"],
      "자전거": ["bicycle", "riding a bike", "cycling"],
      "해변/바다": ["beach", "ocean", "the sea"],
      "눈/산": ["snow", "mountain", "ski"],
      "자동차": ["a car ", "cars on", "vehicle on the road"],
      "꽃": ["flower", "flowers in"],
      "기차": ["train on", "railway", "a train"],
      "보트/배": ["boat", "ship on", "sailing"],
      "아기/아이": ["a baby", "a child", "kids playing"],
      "비행기": ["airplane", "aircraft", "a plane"]}
seed_for = {}
val_nar = "/tmp/oi_nar_val.jsonl"
if os.path.exists(val_nar):
    with open(val_nar) as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            iid = d.get("image_id"); cap = (d.get("caption") or "").lower()
            if iid not in id2row: continue
            for theme, kws in KW.items():
                if theme not in seed_for and any(k in cap for k in kws):
                    seed_for[theme] = (id2row[iid], d.get("caption"))
            if len(seed_for) == len(KW): break

K = 10
def topk(sim, s):  # top-K EXCLUDING the seed itself (i2i rank1 is trivially self)
    o = np.argpartition(-sim, K + 1)[:K + 1]; o = o[np.argsort(-sim[o])]
    return [int(i) for i in o if int(i) != s][:K]

res = []
allp = set()
for theme, (s, cap) in seed_for.items():
    ftop = topk(embn @ embn[s], s)
    btop = topk(cb @ cb[s], s)
    ctop = topk(cc @ cc[s], s)
    fset = set(ftop)
    entry = {"theme": theme, "caption": cap, "seed": paths[s],
             "float": [paths[i] for i in ftop],
             "bit_base": [paths[i] for i in btop],
             "bit_cc12m": [paths[i] for i in ctop],
             "ov_base": round(len(set(btop) & fset) / K * 100),
             "ov_cc12m": round(len(set(ctop) & fset) / K * 100)}
    res.append(entry)
    allp.update([paths[s]] + entry["float"] + entry["bit_base"] + entry["bit_cc12m"])

json.dump(res, open("/tmp/gallery.json", "w"))
with open("/tmp/gallery_imglist.txt", "w") as f:
    for p in sorted(allp):
        f.write(p + "\n")
print(f"GALLERY_DONE: {len(res)} themes, {len(allp)} unique images -> /tmp/gallery.json", flush=True)
for e in res:
    print(f"  {e['theme']}: float-overlap base {e['ov_base']}% / cc12m {e['ov_cc12m']}%", flush=True)
