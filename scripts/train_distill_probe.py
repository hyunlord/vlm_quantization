"""Prototype: does emb-similarity DISTILLATION make the hash preserve SigLIP2
semantics (close the A∩B gap)?  Trains baseline vs distilled heads on cached
COCO embeddings, evals COCO test R@10, then probes OI cross-modal queries.

Distillation loss (added to CombinedHashLoss): within each batch, the hash
continuous similarity matrix must match the teacher emb similarity matrix —
intra-image, intra-text, and cross-modal. This counters InfoNCE pushing apart
same-class items (e.g. other Labradors) that emb knows are similar.

Env: BITS(8..256), EPOCHS(30), DISTILL_W(3.0), OI_INDEX
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "30")); BS = 512
DISTILL_W = float(os.environ.get("DISTILL_W", "3.0"))
OI_INDEX = os.environ.get("OI_INDEX", "/tmp/oi_index_167k.npz")
MODEL = "google/siglip2-so400m-patch14-384"
dev = "cuda" if torch.cuda.is_available() else "cpu"
MB = BITS[-1]
KS = [1, 5, 10]

P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
clean, weak, strong, txt = TC["clean"], TC["weak"], TC["strong"], TC["txt"]
te_i, te_t, labels = EC["test"]["img"], EC["test"]["txt"], EC["test"]["ids"]
embed, N = clean.shape[1], clean.shape[0]
print(f"train N={N} embed={embed} | BITS={BITS} EPOCHS={EPOCHS} DISTILL_W={DISTILL_W}", flush=True)


def eval_rank(scores, larger):
    order = scores.argsort(dim=1, descending=larger)
    rel = (labels[order] == labels[:, None]).float()
    return {f"R@{k}": round((rel[:, :k].sum(1) > 0).float().mean().item(), 4) for k in KS}


def hdist(q, db):
    return (q.size(1) - q @ db.t()) / 2


def distill(io, to_, ci, ti):
    ic = F.normalize(io[-1]["continuous"], dim=1); tc = F.normalize(to_[-1]["continuous"], dim=1)
    ie = F.normalize(ci, dim=1); te = F.normalize(ti, dim=1)
    return (F.mse_loss(ic @ ic.t(), ie @ ie.t())
            + F.mse_loss(tc @ tc.t(), te @ te.t())
            + F.mse_loss(ic @ tc.t(), ie @ te.t())) / 3.0


def train_head(distill_w):
    torch.manual_seed(42)
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"],
                          quantization_weight=P["quant"], balance_weight=P["balance"],
                          consistency_weight=P["cons"], lcs_weight=P["lcs"], temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = perm[s:s + BS]
            ci, wi, si, ti = clean[idx].to(dev), weak[idx].to(dev), strong[idx].to(dev), txt[idx].to(dev)
            io, to_ = img_h(ci), txt_h(ti)
            out = lf(io, to_, weak_image_outputs=img_h(wi), aug_image_outputs=img_h(si), progress=g / max(steps, 1))
            total = out["total"]
            if distill_w > 0:
                total = total + distill_w * distill(io, to_, ci, ti)
            opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    with torch.no_grad():
        io_, to_ = img_h(te_i.to(dev)), txt_h(te_t.to(dev))
    ie, ten = F.normalize(te_i, dim=1), F.normalize(te_t, dim=1)
    q = {"float_I2T": eval_rank(ie @ ten.T, True),
         f"{MB}bit_I2T": eval_rank(hdist(io_[-1]["binary"].cpu().float(), to_[-1]["binary"].cpu().float()), False)}
    print(f"  [distill_w={distill_w}] trained {time.perf_counter()-t0:.0f}s | COCO {q}", flush=True)
    return img_h, txt_h, q


print("=== training baseline (no distill) ===", flush=True)
ib, tb, qb = train_head(0.0)
print("=== training distilled ===", flush=True)
idl, tdl, qd = train_head(DISTILL_W)

# ---- OI cross-modal probe ----
print("\n=== loading SigLIP2 text + OI image embeddings ===", flush=True)
from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
try: del bb.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
oi = np.ascontiguousarray(np.load(OI_INDEX, allow_pickle=True)["emb"].astype(np.float32))
oi_t = torch.from_numpy(oi)
QUERIES = ["running lab", "a labrador retriever dog running", "a dog running in grass", "library bookshelves"]
K = 12

def img_logits(img_h):
    out = np.empty((len(oi), MB), np.float32)
    with torch.no_grad():
        for i in range(0, len(oi), 8192):
            out[i:i+8192] = img_h(oi_t[i:i+8192].to(dev))[-1]["continuous"].cpu().numpy()
    return out

def text_logits(txt_h, q):
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e = pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                 attention_mask=t.get("attention_mask").to(dev) if t.get("attention_mask") is not None else None)).float()
        tl = txt_h(e.to(dev))[-1]["continuous"][0].cpu().numpy()
    return F.normalize(e, dim=1)[0].cpu().numpy(), tl

def topk(sc): return set(np.argpartition(-sc, K-1)[:K].tolist())

print(f"\n=== OI cross-modal A∩B (hash preserves emb ranking?), k={K} ===")
print(f"{'query':36} | baseline A∩B | distill A∩B")
print("-"*72)
oi_norm = oi / (np.linalg.norm(oi, axis=1, keepdims=True) + 1e-9)
ilog_b, ilog_d = img_logits(ib), img_logits(idl)
for q in QUERIES:
    te_b, tl_b = text_logits(tb, q); A_b = topk(oi_norm @ te_b); B_b = topk(ilog_b @ tl_b)
    te_d, tl_d = text_logits(tdl, q); A_d = topk(oi_norm @ te_d); B_d = topk(ilog_d @ tl_d)
    print(f"{q:36} | {len(A_b&B_b)/K*100:9.0f}%  | {len(A_d&B_d)/K*100:9.0f}%")
print(f"\nCOCO test R@10 (I2T):  baseline float {qb['float_I2T']['R@10']} / {MB}bit {qb[f'{MB}bit_I2T']['R@10']}"
      f"  |  distill float {qd['float_I2T']['R@10']} / {MB}bit {qd[f'{MB}bit_I2T']['R@10']}")
