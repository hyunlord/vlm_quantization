"""Out-of-distribution benchmark #2: DOCCI test (5000 images, 1 dense caption each;
NEVER trained on). float vs bit hash heads. 1:1 protocol (gold = paired item).
DOCCI captions are long/dense -> truncated to 64 tokens (same for float & bit).
"""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
MODEL = "google/siglip2-so400m-patch14-384"
ROOT = f"{REPO}/data/docci"; JL = f"{ROOT}/docci_test.jsonl"; IMGDIR = f"{ROOT}/images"
dev = "cuda" if torch.cuda.is_available() else "cpu"; CACHE = "/tmp/docci_emb.pt"; KS = [1, 5, 10]

entries = [json.loads(l) for l in open(JL)]
img_paths = [e["image_path"] for e in entries]
caps = [e["caption"] for e in entries]
N = len(entries); cap2img = torch.arange(N)
print(f"DOCCI test: {N} images (1:1 dense caption)", flush=True)

if os.path.exists(CACHE):
    d = torch.load(CACHE, map_location="cpu"); I = d["img"]; T = d["txt"]; print("loaded cache", flush=True)
else:
    from transformers import AutoModel, SiglipImageProcessor
    ip = SiglipImageProcessor.from_pretrained(MODEL)
    try:
        from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
    except Exception:
        from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
    bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
    with torch.no_grad():
        Iv = []
        for s in range(0, N, 64):
            ims = [Image.open(f"{IMGDIR}/{p}").convert("RGB") for p in img_paths[s:s+64]]
            px = ip(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            Iv.append(pool(bb.vision_model(pixel_values=px)).float().cpu())
            if s % 1024 == 0: print(f"  img {s}/{N}", flush=True)
        I = torch.cat(Iv, 0)
        Tv = []
        for s in range(0, N, 256):
            t = tok(caps[s:s+256], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            am = t.get("attention_mask")
            Tv.append(pool(bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)).float().cpu())
        T = torch.cat(Tv, 0)
    torch.save({"img": I, "txt": T}, CACHE); print("embedded + cached", flush=True)

In = F.normalize(I.float(), dim=1); Tn = F.normalize(T.float(), dim=1)
def rk_t2i(sim):
    order = sim.argsort(1, descending=True); rank = (order == cap2img[:, None]).float().argmax(1)
    return {f"R@{k}": round((rank < k).float().mean().item()*100, 2) for k in KS}
def rk_i2t(sim):
    order = sim.argsort(1, descending=True); rel = (cap2img[order] == torch.arange(N)[:, None])
    return {f"R@{k}": round((rel[:, :k].any(1)).float().mean().item()*100, 2) for k in KS}
def hsim(qa, qb): return qa @ qb.t()
def head_codes(path, bit=1024):
    h = torch.load(path, map_location="cpu"); b = h["bits"]; bi = b.index(bit) if bit in b else len(b)-1
    ih = NestedHashLayer(h["embed"], h["hidden"], b, 0.1); ih.load_state_dict(h["img_h"]); ih.eval()
    th = NestedHashLayer(h["embed"], h["hidden"], b, 0.1); th.load_state_dict(h["txt_h"]); th.eval()
    with torch.no_grad(): return ih(In)[bi]["binary"].float(), th(Tn)[bi]["binary"].float()
RUNS = [("float (ceiling)", None), ("best 1:1 @1024", "/tmp/sweep_c113287_coco_oi_rkd_crovca.pt"),
        ("best 1:2 @1024", "/tmp/sweep_c226574_coco_oi_rkd_crovca.pt"), ("ft113 (1:1+koft)", "/tmp/ft_ko_113.pt"),
        ("ft226 (1:2+koft)", "/tmp/ft_ko_226.pt"), ("k12b113 (1.38M best)", "/tmp/k12b113_coco_oi_rkd_crovca.pt"),
        ("ft12_113 (1.38M+koft)", "/tmp/ft12_113.pt"),
        ("CC12M 403K @1024", "/tmp/k1024_coco_oi_rkd_crovca.pt"), ("COCO baseline @1024", "/tmp/k1024_coco.pt")]
print(f"\n{'setting':22} | I2T R@1/5/10        | T2I R@1/5/10", flush=True); print("-"*70)
for label, path in RUNS:
    if path is None: i2t = rk_i2t(In @ Tn.t()); t2i = rk_t2i(Tn @ In.t())
    else:
        ic, tc = head_codes(path); i2t = rk_i2t(hsim(ic, tc)); t2i = rk_t2i(hsim(tc, ic))
    print(f"{label:22} | {i2t['R@1']:5}/{i2t['R@5']:5}/{i2t['R@10']:5} | {t2i['R@1']:5}/{t2i['R@5']:5}/{t2i['R@10']:5}", flush=True)
print("DOCCI_BENCH_DONE", flush=True)
