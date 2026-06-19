"""CLEAN ablation: hash-head INPUT normalization on vs off, all else identical.

Controlled counterfactual for the claim "raw input -> 'running lab'=library;
normalized input -> dogs". COCO-only, seed=42, same HP/epochs/BITS as bit-base
(round-4 broaden_head_coco). The ONLY difference across the two runs is whether
the hash head receives L2-normalized or raw pooled embeddings.

  - float baseline (gold) ALWAYS uses normalized cosine (fixed reference).
  - loss-internal normalizations (none here; mode=coco) untouched.
  - probe: same OI text->image t2i + agreement@100 as train_broaden.

Forced CPU (GPU busy with CC12M embed). Env: NORM_IN(1|0), TAG.
"""
from __future__ import annotations
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only — do not contend with embed
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

NORM_IN = int(os.environ.get("NORM_IN", "1"))
TAG = os.environ.get("TAG", "norm" if NORM_IN else "raw")
BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "25")); BS = 512
OI_INDEX = os.environ.get("OI_INDEX", "/tmp/oi_index_167k.npz")
MODEL = "google/siglip2-so400m-patch14-384"
dev = "cpu"; MB = BITS[-1]; KS = [1, 5, 10]

def nrm(x): return F.normalize(x, dim=1)
def inp(x): return nrm(x) if NORM_IN else x   # hash-head INPUT (the toggle)

P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")

# hash-head inputs (toggled)
clean, weak, strong, txt = inp(TC["clean"]), inp(TC["weak"]), inp(TC["strong"]), inp(TC["txt"])
te_i, te_t = inp(EC["test"]["img"]), inp(EC["test"]["txt"])
# float gold baseline (always normalized cosine)
te_i_n, te_t_n = nrm(EC["test"]["img"]), nrm(EC["test"]["txt"])
labels = EC["test"]["ids"]
embed, N = clean.shape[1], clean.shape[0]
print(f"[{TAG}] NORM_IN={NORM_IN} COCO N={N} BITS={BITS} EPOCHS={EPOCHS}", flush=True)


def eval_rank(scores, larger):
    order = scores.argsort(dim=1, descending=larger)
    rel = (labels[order] == labels[:, None]).float()
    return {f"R@{k}": round((rel[:, :k].sum(1) > 0).float().mean().item(), 4) for k in KS}


def hdist(q, db): return (q.size(1) - q @ db.t()) / 2


def train():
    torch.manual_seed(42)
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    spc = N // BS; steps = EPOCHS * spc
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        pc = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = pc[s:s+BS]
            ci, wi, si, ti = clean[idx], weak[idx], strong[idx], txt[idx]
            io = img_h(ci)
            out = lf(io, txt_h(ti), weak_image_outputs=img_h(wi), aug_image_outputs=img_h(si), progress=g/max(steps,1))
            opt.zero_grad(); out["total"].backward(); opt.step(); sched.step(); g += 1
        if ep % 5 == 0:
            print(f"  [{TAG}] ep{ep} {time.perf_counter()-t0:.0f}s", flush=True)
    img_h.eval(); txt_h.eval()
    with torch.no_grad():
        io_, to_ = img_h(te_i), txt_h(te_t)
    q = {"float": eval_rank(te_i_n @ te_t_n.T, True),
         f"{MB}bit": eval_rank(hdist(io_[-1]["binary"].float(), to_[-1]["binary"].float()), False)}
    print(f"  [{TAG}] DONE {time.perf_counter()-t0:.0f}s | float R@10 {q['float']['R@10']} | {MB}bit R@10 {q[f'{MB}bit']['R@10']}", flush=True)
    torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(),
                "bits": BITS, "hidden": P["hidden"], "embed": embed, "norm_in": NORM_IN},
               f"/tmp/ablate_head_{TAG}.pt")
    return img_h, txt_h, q


img_h, txt_h, q = train()

# ---- t2i probe + agreement@100 (same protocol as train_broaden) ----
print(f"\n[{TAG}] loading SigLIP2 text + OI image emb for probe", flush=True)
from transformers import AutoModel
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
try: del bb.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

oi_raw = np.ascontiguousarray(np.load(OI_INDEX, allow_pickle=True)["emb"].astype(np.float32))
oi_norm = oi_raw / (np.linalg.norm(oi_raw, axis=1, keepdims=True) + 1e-9)
oi_in = torch.from_numpy(oi_norm if NORM_IN else oi_raw)   # img_h corpus input (toggled)
oi_cos = torch.from_numpy(oi_norm)                         # float gold corpus (normalized)

def img_logits(h):
    out = np.empty((len(oi_in), MB), np.float32)
    with torch.no_grad():
        for i in range(0, len(oi_in), 8192):
            out[i:i+8192] = h(oi_in[i:i+8192])[-1]["continuous"].numpy()
    return out

QUERIES = ["running lab", "a labrador retriever dog running", "a dog running in grass",
           "library bookshelves", "a person riding a horse", "a slice of pizza", "a steam train"]
K = 12
def topk(sc): return set(np.argpartition(-sc, K-1)[:K].tolist())

def tlog(q):
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e = pool(bb.text_model(input_ids=t["input_ids"],
                 attention_mask=t.get("attention_mask"))).float()
        head_in = nrm(e) if NORM_IN else e
        tl = txt_h(head_in)[-1]["continuous"][0].numpy()
    return nrm(e)[0].numpy(), tl   # (float-cosine query[normalized], hash text logits)

ilog = img_logits(img_h)
print(f"\n[{TAG}] OI t2i A(float)∩B(hash), k={K}")
for qq in QUERIES:
    te, tl = tlog(qq)
    A = topk(oi_cos.numpy() @ te); B = topk(ilog @ tl)
    print(f"  {qq:34} | overlap {len(A&B)/K*100:3.0f}% | hashtop_caps_idx {sorted(list(B))[:4]}", flush=True)

# agreement@100 (model-free query captions from OI text emb)
oi_txt_pairs = torch.load("/tmp/oi_pairs.pt", map_location="cpu")["txt_emb"]
oi_txt_in = inp(oi_txt_pairs)
qidx = np.random.default_rng(0).choice(oi_txt_in.shape[0], size=min(200, oi_txt_in.shape[0]), replace=False)
qsub = oi_txt_in[qidx]
es = oi_norm @ nrm(oi_txt_pairs)[qidx].numpy().T
emb_top = [set(np.argpartition(-es[:, j], 99)[:100].tolist()) for j in range(qsub.shape[0])]
with torch.no_grad():
    tq = txt_h(qsub)[-1]["continuous"].numpy()
hs = ilog @ tq.T
ag = np.mean([len(set(np.argpartition(-hs[:, j], 99)[:100].tolist()) & emb_top[j]) / 100 for j in range(qsub.shape[0])])
print(f"\n[{TAG}] agreement@100 = {ag*100:.1f}%  (COCO {MB}bit R@10 {q[f'{MB}bit']['R@10']})", flush=True)
print(f"ABLATE_DONE_{TAG}", flush=True)
