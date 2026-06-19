"""Out-of-distribution generalization benchmark: Flickr30K-1K test (NEVER trained
on — we trained on COCO+CC12M+OI). float (raw SigLIP2) vs our bit hash heads.

5-caption protocol (1000 imgs x 5 caps = 5000 caps):
  T2I: each caption ranks 1000 images; R@k = its source image in top-k.
  I2T: each image ranks 5000 captions; R@k = ANY of its 5 captions in top-k.
Embeds Flickr once (cache /tmp/flickr_emb.pt), then float + bit (1024-bit) for
several heads. Compares to the COCO (in-domain) numbers to show generalization.
"""
import sys, os, io, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
MODEL = "google/siglip2-so400m-patch14-384"
ROOT = f"{REPO}/data/flickr30k"; JL = f"{ROOT}/flickr30k_test.jsonl"
dev = "cuda" if torch.cuda.is_available() else "cpu"; CACHE = "/tmp/flickr_emb.pt"
KS = [1, 5, 10]

entries = [json.loads(l) for l in open(JL)]
img_paths = [e["image_path"] for e in entries]
caps, cap2img = [], []
for i, e in enumerate(entries):
    for c in e["captions"][:5]: caps.append(c); cap2img.append(i)
cap2img = torch.tensor(cap2img); NI = len(img_paths); NC = len(caps)
print(f"Flickr30K test: {NI} images, {NC} captions", flush=True)

if os.path.exists(CACHE):
    d = torch.load(CACHE, map_location="cpu"); I = d["img"]; T = d["txt"]
    print("loaded cached Flickr emb", flush=True)
else:
    from transformers import AutoModel, SiglipImageProcessor
    ip = SiglipImageProcessor.from_pretrained(MODEL)
    try:
        from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
    except Exception:
        from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
    bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
    Iv = []
    with torch.no_grad():
        for s in range(0, NI, 64):
            ims = [Image.open(f"{ROOT}/{p}").convert("RGB") for p in img_paths[s:s+64]]
            px = ip(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            Iv.append(pool(bb.vision_model(pixel_values=px)).float().cpu())
            print(f"  img {min(s+64,NI)}/{NI}", flush=True)
        I = torch.cat(Iv, 0)
        Tv = []
        for s in range(0, NC, 256):
            t = tok(caps[s:s+256], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            am = t.get("attention_mask")
            Tv.append(pool(bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)).float().cpu())
        T = torch.cat(Tv, 0)
    torch.save({"img": I, "txt": T}, CACHE); print("embedded + cached", flush=True)

In = F.normalize(I.float(), dim=1); Tn = F.normalize(T.float(), dim=1)

def rk_t2i(sim):  # sim (NC, NI): caption rows; gold = cap2img
    order = sim.argsort(1, descending=True)
    rank = (order == cap2img[:, None]).float().argmax(1)
    return {f"R@{k}": round((rank < k).float().mean().item()*100, 2) for k in KS}
def rk_i2t(sim):  # sim (NI, NC): image rows; gold = any caption with cap2img==i
    order = sim.argsort(1, descending=True)
    rel = (cap2img[order] == torch.arange(NI)[:, None])      # (NI,NC) bool
    return {f"R@{k}": round((rel[:, :k].any(1)).float().mean().item()*100, 2) for k in KS}

def hsim(qa, qb):  # similarity = -hamming = (q@db - bit)/... use dot (higher=closer)
    return qa @ qb.t()   # ±1 codes: larger dot = smaller hamming -> descending works

def head_codes(path, bit=1024):
    h = torch.load(path, map_location="cpu"); b = h["bits"]; bi = b.index(bit) if bit in b else len(b)-1
    ih = NestedHashLayer(h["embed"], h["hidden"], b, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], b, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    with torch.no_grad():
        return ih(In)[bi]["binary"].float(), th(Tn)[bi]["binary"].float()

RUNS = [
    ("float (ceiling)", None),
    ("best 1:1 @1024", "/tmp/sweep_c113287_coco_oi_rkd_crovca.pt"),
    ("best 1:2 @1024", "/tmp/sweep_c226574_coco_oi_rkd_crovca.pt"),
    ("ft113 (1:1+koft)", "/tmp/ft_ko_113.pt"),
    ("ft226 (1:2+koft)", "/tmp/ft_ko_226.pt"),
    ("k12b113 (1.38M best)", "/tmp/k12b113_coco_oi_rkd_crovca.pt"),
    ("ft12_113 (1.38M+koft)", "/tmp/ft12_113.pt"),
    ("CC12M 403K @1024", "/tmp/k1024_coco_oi_rkd_crovca.pt"),
    ("COCO baseline @1024", "/tmp/k1024_coco.pt"),
]
print(f"\n{'setting':22} | I2T R@1/5/10        | T2I R@1/5/10", flush=True)
print("-"*70)
for label, path in RUNS:
    if path is None:
        i2t = rk_i2t(In @ Tn.t()); t2i = rk_t2i(Tn @ In.t())
    else:
        ic, tc = head_codes(path); i2t = rk_i2t(hsim(ic, tc)); t2i = rk_t2i(hsim(tc, ic))
    print(f"{label:22} | {i2t['R@1']:5}/{i2t['R@5']:5}/{i2t['R@10']:5} | {t2i['R@1']:5}/{t2i['R@5']:5}/{t2i['R@10']:5}", flush=True)
print("FLICKR_BENCH_DONE", flush=True)
