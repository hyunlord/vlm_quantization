"""GPU-FREE text->image (t2i) galleries — 4-way: float / old-demo / bit-base / bit-CC12M.

Each method uses its AS-TRAINED convention:
  float       : normalized text_emb . normalized image_emb (gold)
  bit-old     : OLD demo head (demo_hashheads.pt), RAW input (as trained);
                corpus codes = the demo's stored packed_256 in oi_index
  bit-base    : round-4 coco head, NORMALIZED input
  bit-CC12M   : round-4 CC12M head, NORMALIZED input
Text encoder forced to CPU -> no GPU contention. Out: /tmp/gallery_t2i.json + imglist.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, torch, torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
MODEL = "google/siglip2-so400m-patch14-384"

idx = np.load("/tmp/oi_index_167k.npz", allow_pickle=True)
emb = np.ascontiguousarray(idx["emb"].astype(np.float32))   # already L2-normalized
paths = [str(p) for p in idx["paths"]]
N = len(emb)

def load_txt_img(p):
    hh = torch.load(p, map_location="cpu")
    bi = hh["bits"].index(256) if 256 in hh["bits"] else len(hh["bits"]) - 1
    ih = NestedHashLayer(hh["embed"], hh["hidden"], hh["bits"], 0.1); ih.load_state_dict(hh["img_h"]); ih.eval()
    th = NestedHashLayer(hh["embed"], hh["hidden"], hh["bits"], 0.1); th.load_state_dict(hh["txt_h"]); th.eval()
    return ih, th, bi

ib, tb, bi = load_txt_img("/tmp/broaden_head_coco.pt")          # NEW baseline (normalized)
ic, tc, _ = load_txt_img("/tmp/broaden_head_coco_oi_rkd_crovca.pt")  # NEW CC12M (normalized)

# OLD demo: use its txt head (raw input) + the demo's STORED image codes (packed_256)
dhh = torch.load("/tmp/demo_hashheads.pt", map_location="cpu")
dbi = dhh["bits"].index(256)
dth = NestedHashLayer(dhh["embed"], dhh["hidden"], dhh["bits"], 0.1); dth.load_state_dict(dhh["txt_h"]); dth.eval()
cold = (np.unpackbits(idx["packed_256"].astype(np.uint8), axis=1).astype(np.float32) * 2 - 1)  # (N,256) ±1

def img_codes(h):  # NEW heads: corpus codes from NORMALIZED emb (matches their training)
    out = []
    with torch.no_grad():
        for i in range(0, N, 16384):
            out.append(h(torch.from_numpy(emb[i:i+16384]))[bi]["binary"].numpy().astype(np.float32))
    return np.concatenate(out, 0)
print("corpus codes (cpu)...", flush=True)
cb, cc = img_codes(ib), img_codes(ic)

from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
try: del bb.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

QUERIES = ["running lab", "a labrador retriever running on grass", "library full of books",
           "a slice of pizza", "birthday cake with candles", "a person riding a horse",
           "red umbrella in the rain", "a cat sleeping on a couch", "snowy mountain landscape",
           "a sailboat on the ocean", "a steam train", "a baby playing with toys"]
K = 10
def topk(sim):
    o = np.argpartition(-sim, K)[:K]; return [int(i) for i in o[np.argsort(-sim[o])]]

res = []; allp = set()
for q in QUERIES:
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e = pool(bb.text_model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"))).float()
        en = F.normalize(e, dim=1)
        qn = en[0].numpy()                                   # float / new heads use normalized
        qcb = np.sign(tb(en)[bi]["binary"][0].numpy()).astype(np.float32)
        qcc = np.sign(tc(en)[bi]["binary"][0].numpy()).astype(np.float32)
        qold = np.sign(dth(e)[dbi]["binary"][0].numpy()).astype(np.float32)   # OLD: raw input
    ft = topk(emb @ qn); ot = topk(cold @ qold); btt = topk(cb @ qcb); ct = topk(cc @ qcc)
    fset = set(ft)
    entry = {"query": q, "float": [paths[i] for i in ft], "bit_old": [paths[i] for i in ot],
             "bit_base": [paths[i] for i in btt], "bit_cc12m": [paths[i] for i in ct],
             "ov_old": round(len(set(ot) & fset)/K*100), "ov_base": round(len(set(btt) & fset)/K*100),
             "ov_cc12m": round(len(set(ct) & fset)/K*100)}
    res.append(entry); allp.update(entry["float"] + entry["bit_old"] + entry["bit_base"] + entry["bit_cc12m"])
    print(f"  {q}: old {entry['ov_old']}% base {entry['ov_base']}% cc12m {entry['ov_cc12m']}%", flush=True)

json.dump(res, open("/tmp/gallery_t2i.json", "w"))
open("/tmp/gallery_t2i_imglist.txt", "w").write("\n".join(sorted(allp)) + "\n")
print(f"T2I_GALLERY_DONE: {len(res)} queries, {len(allp)} images", flush=True)
