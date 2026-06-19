"""Query-robustness benchmark: are binary hash codes more robust to query
perturbations (typos / word-dropout / truncation) than the float baseline?

COCO test 5K, text->image. For M sampled captions, encode clean + 3 perturbed
variants, retrieve top-10 over the 5K test-image corpus for float / bit-256 /
bit-1024 (CC12M-403K head). Report, per method x perturbation:
  Stability@10 = |top10(clean) ∩ top10(perturbed)| / 10   (higher = more robust)
  R@10 clean / R@10 perturbed / retention = R@10_pert / R@10_clean
gold = the paired image (1:1). Fixed seed -> deterministic.
"""
import sys, os, json, random
import numpy as np, torch, torch.nn.functional as F
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
MODEL = "google/siglip2-so400m-patch14-384"
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = 10; M = 300
random.seed(0); np.random.seed(0); torch.manual_seed(0)

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
ti = F.normalize(EC["test"]["img"].float(), dim=1)                 # (5000,1152) corpus (float)
ids = EC["test"]["ids"]; idlist = (ids.tolist() if torch.is_tensor(ids) else list(ids))
N = ti.shape[0]
cap = {}
for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]:
    cap[im["cocoid"]] = im["sentences"][0]["raw"] if im.get("sentences") else ""

# head (CC12M 403K) -> corpus bit codes + query encoder
h = torch.load("/tmp/k1024_coco_oi_rkd_crovca.pt", map_location="cpu"); bits = h["bits"]
ih = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); ih.load_state_dict(h["img_h"]); ih.to(dev).eval()
th = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); th.load_state_dict(h["txt_h"]); th.to(dev).eval()
ccodes = {}
with torch.no_grad():
    o = ih(ti.to(dev))
    for b in (256, 1024): ccodes[b] = o[bits.index(b)]["binary"].cpu()      # (N,b) ±1
from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
try: del bb.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

def enc(text):
    t = tok([text], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    am = t.get("attention_mask")
    with torch.no_grad():
        e = pool(bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)).float()
        en = F.normalize(e, dim=1)
        qb = {b: torch.sign(th(en)[bits.index(b)]["binary"][0]).cpu() for b in (256, 1024)}
    return en[0].cpu(), qb

# ---- perturbations ----
def typo(s, p=0.15):
    out = []
    for ch in s:
        if ch != " " and random.random() < p:
            op = random.choice(["drop", "rep", "ins", "swap"])
            if op == "drop": continue
            if op == "rep": out.append(random.choice("abcdefghijklmnopqrstuvwxyz")); continue
            if op == "ins": out.append(ch); out.append(random.choice("abcdefghijklmnopqrstuvwxyz")); continue
            if op == "swap" and out: out.append(ch); out[-1], out[-2] = out[-2], out[-1]; continue
        out.append(ch)
    return "".join(out) or s
def worddrop(s, p=0.3):
    w = [x for x in s.split() if random.random() > p]
    return " ".join(w) or s
def trunc(s):
    w = s.split(); return " ".join(w[:max(1, int(len(w) * 0.6))])
PERTS = {"typo": typo, "worddrop": worddrop, "trunc": trunc}

def topk_float(fq): s = ti @ fq; o = np.argpartition(-s.numpy(), K)[:K]; return set(int(x) for x in o)
def topk_bit(qc, b): d = (b - (ccodes[b] @ qc)) / 2; o = np.argpartition(d.numpy(), K)[:K]; return set(int(x) for x in o)
def topk(method, fq, qb):
    return topk_float(fq) if method == "float" else topk_bit(qb[256], 256) if method == "bit256" else topk_bit(qb[1024], 1024)

rows = [i for i in range(N) if cap.get(idlist[i])][:M]
print(f"queries={len(rows)} K={K}", flush=True)
METHODS = ["float", "bit256", "bit1024"]
stab = {m: {p: [] for p in PERTS} for m in METHODS}
r10c = {m: [] for m in METHODS}; r10p = {m: {p: [] for p in PERTS} for m in METHODS}

for i in rows:
    text = cap[idlist[i]]
    fq_c, qb_c = enc(text)
    tk_c = {m: topk(m, fq_c, qb_c) for m in METHODS}
    for m in METHODS: r10c[m].append(1.0 if i in tk_c[m] else 0.0)
    for pn, pf in PERTS.items():
        fq_p, qb_p = enc(pf(text))
        for m in METHODS:
            tk_p = topk(m, fq_p, qb_p)
            stab[m][pn].append(len(tk_c[m] & tk_p) / K)
            r10p[m][pn].append(1.0 if i in tk_p else 0.0)

print("\n=== Stability@10 (top10 clean∩perturbed /10; higher=robust) ===", flush=True)
print(f"{'method':9} | " + " | ".join(f"{p:>9}" for p in PERTS))
for m in METHODS:
    print(f"{m:9} | " + " | ".join(f"{np.mean(stab[m][p])*100:8.1f}%" for p in PERTS))
print("\n=== R@10 clean -> perturbed (retention%) ===", flush=True)
print(f"{'method':9} | {'clean':>7} | " + " | ".join(f"{p:>16}" for p in PERTS))
for m in METHODS:
    c = np.mean(r10c[m]) * 100
    cells = []
    for p in PERTS:
        pv = np.mean(r10p[m][p]) * 100; cells.append(f"{pv:5.1f} ({pv/max(c,1e-9)*100:3.0f}%)")
    print(f"{m:9} | {c:6.1f}% | " + " | ".join(f"{x:>16}" for x in cells))
print("ROBUSTNESS_DONE", flush=True)
