"""Data-broadening + RKD experiment (research-informed).

Compares 3 training conditions for the hash head (frozen backbone, cached emb):
  coco        : COCO only (baseline)
  coco_oi     : COCO + OI image-text pairs (Localized Narratives) — DATA broadening
  coco_oi_rkd : + RKD relational distillation (distance-wise, mu-normalized, Huber,
                low weight) on image structure — the *correct* distillation
                (relative/scale-invariant, InfoNCE-compatible) vs our failed absolute L2.

All hash-head inputs are L2-normalized (unifies COCO raw + OI normalized emb).
Eval: COCO test I2T R@10  +  OI cross-modal A∩B probe ("running lab" etc.).

Env: BITS(8..256), EPOCHS(25), W_RKD(0.3), OI_PAIRS, OI_INDEX, MODES
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "25")); BS = 512
W_RKD = float(os.environ.get("W_RKD", "0.3"))
W_CROVCA = float(os.environ.get("W_CROVCA", "0.1"))
OI_PAIRS = os.environ.get("OI_PAIRS", "/tmp/oi_pairs.pt")
OI_INDEX = os.environ.get("OI_INDEX", "/tmp/oi_index_167k.npz")
MODES = os.environ.get("MODES", "coco,coco_oi,coco_oi_rkd").split(",")
MODEL = "google/siglip2-so400m-patch14-384"
dev = "cuda" if torch.cuda.is_available() else "cpu"
MB = BITS[-1]; KS = [1, 5, 10]

P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
def nrm(x): return F.normalize(x, dim=1)
clean = nrm(TC["clean"]); weak = nrm(TC["weak"]); strong = nrm(TC["strong"]); txt = nrm(TC["txt"])
te_i = nrm(EC["test"]["img"]); te_t = nrm(EC["test"]["txt"]); labels = EC["test"]["ids"]
embed, N = clean.shape[1], clean.shape[0]
OI = torch.load(OI_PAIRS, map_location="cpu")
oi_img, oi_txt = nrm(OI["img_emb"]), nrm(OI["txt_emb"]); NO = oi_img.shape[0]
SMOKE = int(os.environ.get("SMOKE", "0"))
if SMOKE:   # code-path validation: tiny slices, real run uses SMOKE=0
    clean, weak, strong, txt = clean[:SMOKE], weak[:SMOKE], strong[:SMOKE], txt[:SMOKE]
    te_i, te_t, labels = te_i[:SMOKE], te_t[:SMOKE], labels[:SMOKE]
    oi_img, oi_txt = oi_img[:SMOKE], oi_txt[:SMOKE]; N = clean.shape[0]; NO = oi_img.shape[0]
print(f"COCO N={N} | OI pairs={NO} | BITS={BITS} EPOCHS={EPOCHS} W_RKD={W_RKD} MODES={MODES}", flush=True)


def eval_rank(scores, larger):
    order = scores.argsort(dim=1, descending=larger)
    rel = (labels[order] == labels[:, None]).float()
    return {f"R@{k}": round((rel[:, :k].sum(1) > 0).float().mean().item(), 4) for k in KS}


def hdist(q, db): return (q.size(1) - q @ db.t()) / 2


def rkd_dist(student_cont, teacher_emb):
    """Relational KD, distance-wise: mu-normalized pairwise distances, Huber.
    Scale-invariant -> compatible with InfoNCE (unlike absolute-sim L2)."""
    s = F.normalize(student_cont, dim=1)
    with torch.no_grad():
        t = F.normalize(teacher_emb, dim=1)
        td = torch.cdist(t, t); td = td / (td[td > 0].mean() + 1e-8)
    sd = torch.cdist(s, s); sd = sd / (sd[sd > 0].mean() + 1e-8)
    return F.huber_loss(sd, td)


def coding_rate(z, eps=0.5):
    """CroVCA coding-rate diversity (arXiv 2510.27584), as a loss to MINIMIZE.
    MUST L2-normalize each row first (per-sample), use XᵀX (trace=B, not mean-sub
    covariance), and rescale by (B+d)/(B·d) to keep it O(1) across prefixes —
    verified against the paper (omitting row-norm gave ~30x wrong scale)."""
    z = F.normalize(z, dim=1)
    B, d = z.shape
    cov = z.t() @ z                                   # (d,d), trace = B
    I = torch.eye(d, device=z.device, dtype=z.dtype)
    R = torch.logdet(I + (d / (B * eps)) * cov)       # maximal coding rate
    return -R * (B + d) / (B * d)                     # minimize -> maximize diversity


def train(mode):
    torch.manual_seed(42)
    use_oi = "oi" in mode; use_rkd = "rkd" in mode; use_cr = "crovca" in mode
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    spc = N // BS; spo = (NO // BS) if use_oi else 0
    steps = EPOCHS * (spc + spo)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        pc = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = pc[s:s+BS]
            ci, wi, si, ti = clean[idx].to(dev), weak[idx].to(dev), strong[idx].to(dev), txt[idx].to(dev)
            io = img_h(ci)
            out = lf(io, txt_h(ti), weak_image_outputs=img_h(wi), aug_image_outputs=img_h(si), progress=g/max(steps,1))
            total = out["total"] + (W_RKD * rkd_dist(io[-1]["continuous"], ci) if use_rkd else 0.0) \
                                 + (W_CROVCA * coding_rate(io[-1]["continuous"]) if use_cr else 0.0)
            opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1
        if use_oi:
            po = torch.randperm(NO)
            for s in range(0, NO - BS + 1, BS):
                idx = po[s:s+BS]
                oi_i, oi_t = oi_img[idx].to(dev), oi_txt[idx].to(dev)
                io = img_h(oi_i)
                out = lf(io, txt_h(oi_t), progress=g/max(steps,1))   # InfoNCE only (no aug/labels)
                total = out["total"] + (W_RKD * rkd_dist(io[-1]["continuous"], oi_i) if use_rkd else 0.0) \
                                     + (W_CROVCA * coding_rate(io[-1]["continuous"]) if use_cr else 0.0)
                opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    with torch.no_grad():
        io_, to_ = img_h(te_i.to(dev)), txt_h(te_t.to(dev))
    q = {"float": eval_rank(te_i @ te_t.T, True),
         f"{MB}bit": eval_rank(hdist(io_[-1]["binary"].cpu().float(), to_[-1]["binary"].cpu().float()), False)}
    print(f"  [{mode}] {time.perf_counter()-t0:.0f}s | COCO float R@10 {q['float']['R@10']} | {MB}bit R@10 {q[f'{MB}bit']['R@10']}", flush=True)
    torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(),
                "bits": BITS, "hidden": P["hidden"], "embed": embed, "mode": mode},
               f"/tmp/broaden_head_{mode}.pt")
    return img_h, txt_h, q


heads = {}
for m in MODES:
    print(f"=== training {m} ===", flush=True)
    heads[m] = train(m)

# ---- OI cross-modal A∩B probe ----
print("\n=== loading SigLIP2 text + OI image emb for probe ===", flush=True)
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
oin = oi / (np.linalg.norm(oi, axis=1, keepdims=True) + 1e-9)
if SMOKE:
    oin = oin[:max(SMOKE, 2000)]
oit = torch.from_numpy(oin)
QUERIES = ["running lab", "a labrador retriever dog running", "a dog running in grass", "library bookshelves", "a person riding a horse"]
K = 12
def topk(sc): return set(np.argpartition(-sc, K-1)[:K].tolist())
def ilog(img_h):
    out = np.empty((len(oin), MB), np.float32)
    with torch.no_grad():
        for i in range(0, len(oin), 8192):
            out[i:i+8192] = img_h(oit[i:i+8192].to(dev))[-1]["continuous"].cpu().numpy()
    return out
def tlog(txt_h, q):
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        e = pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                 attention_mask=t.get("attention_mask").to(dev) if t.get("attention_mask") is not None else None)).float()
        tl = txt_h(F.normalize(e, dim=1).to(dev))[-1]["continuous"][0].cpu().numpy()
    return F.normalize(e, dim=1)[0].cpu().numpy(), tl

logits = {m: ilog(heads[m][0]) for m in MODES}

# robust hash<->emb agreement: 200 query captions x k=100 (model-free via oi_txt, low-noise)
qidx = np.random.default_rng(0).choice(oi_txt.shape[0], size=min(200, oi_txt.shape[0]), replace=False)
qsub = oi_txt[qidx]
es = oin @ qsub.numpy().T
emb_top = [set(np.argpartition(-es[:, j], 99)[:100].tolist()) for j in range(qsub.shape[0])]
print(f"\n=== robust hash<->emb agreement@100 ({qsub.shape[0]} query captions, low-noise) ===")
for m in MODES:
    with torch.no_grad():
        tq = heads[m][1](qsub.to(dev))[-1]["continuous"].cpu().numpy()
    hs = logits[m] @ tq.T
    ag = np.mean([len(set(np.argpartition(-hs[:, j], 99)[:100].tolist()) & emb_top[j]) / 100 for j in range(qsub.shape[0])])
    print(f"  {m:22}: {ag*100:.1f}%  (COCO {MB}bit R@10 {heads[m][2][f'{MB}bit']['R@10']})")

print(f"\n=== OI cross-modal A∩B (illustrative hand queries), k={K} ===")
hdr = "query".ljust(34) + " | " + " | ".join(m.center(11) for m in MODES)
print(hdr); print("-"*len(hdr))
for q in QUERIES:
    cells = []
    for m in MODES:
        te, tl = tlog(heads[m][1], q)
        A = topk(oin @ te); B = topk(logits[m] @ tl)
        cells.append(f"{len(A&B)/K*100:9.0f}%")
    print(q.ljust(34) + " | " + " | ".join(c.center(11) for c in cells))
print("\nCOCO R@10:  " + " | ".join(f"{m}: {heads[m][2][f'{MB}bit']['R@10']}" for m in MODES))
