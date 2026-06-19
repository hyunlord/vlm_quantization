"""Mixture-ratio CONTROLLED data experiment.

Removes the round-5/6 confound: COCO loop is fixed 113K while the OI loop grew
with the dataset -> per-epoch non-COCO step ratio drifted. Here the OI loop is
capped to a FIXED per-epoch step budget (OI_CAP samples, re-drawn each epoch),
so only the data POOL changes, not the COCO:OI step ratio.

Also fixes cross-run comparability: the robust agreement@100 query set is loaded
from a FIXED source (/tmp/oi_pairs.pt, seed 0, 200 caps) independent of OI_PAIRS,
so agreement is apples-to-apples across runs (round-3/5/6 sampled queries from
their own training pool -> not comparable).

Env: BITS, EPOCHS(25), W_RKD(1.0), W_CROVCA(0.1), OI_PAIRS, OI_INDEX, OI_CAP(0=off), MODES, TAG.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "25")); BS = 512
W_RKD = float(os.environ.get("W_RKD", "1.0"))
W_CROVCA = float(os.environ.get("W_CROVCA", "0.1"))
OI_PAIRS = os.environ.get("OI_PAIRS", "/tmp/cc12m_pairs.pt")
OI_INDEX = os.environ.get("OI_INDEX", "/tmp/oi_index_167k.npz")
OI_CAP = int(os.environ.get("OI_CAP", "0"))   # per-epoch OI sample budget; 0 = use all
MODES = os.environ.get("MODES", "coco,coco_oi_rkd_crovca").split(",")
TAG = os.environ.get("TAG", "mixctrl")
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
eff_no = min(OI_CAP, NO) if OI_CAP else NO
print(f"[{TAG}] OI_PAIRS={OI_PAIRS} NO={NO} OI_CAP={OI_CAP} -> eff_no={eff_no} | COCO N={N} BITS={BITS} EPOCHS={EPOCHS} MODES={MODES}", flush=True)


def eval_rank(scores, larger):
    order = scores.argsort(dim=1, descending=larger)
    rel = (labels[order] == labels[:, None]).float()
    return {f"R@{k}": round((rel[:, :k].sum(1) > 0).float().mean().item(), 4) for k in KS}


def hdist(q, db): return (q.size(1) - q @ db.t()) / 2


def rkd_dist(student_cont, teacher_emb):
    s = F.normalize(student_cont, dim=1)
    with torch.no_grad():
        t = F.normalize(teacher_emb, dim=1)
        td = torch.cdist(t, t); td = td / (td[td > 0].mean() + 1e-8)
    sd = torch.cdist(s, s); sd = sd / (sd[sd > 0].mean() + 1e-8)
    return F.huber_loss(sd, td)


def coding_rate(z, eps=0.5):
    z = F.normalize(z, dim=1)
    B, d = z.shape
    cov = z.t() @ z
    I = torch.eye(d, device=z.device, dtype=z.dtype)
    R = torch.logdet(I + (d / (B * eps)) * cov)
    return -R * (B + d) / (B * d)


def train(mode):
    torch.manual_seed(42)
    use_oi = "oi" in mode; use_rkd = "rkd" in mode; use_cr = "crovca" in mode
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    spc = N // BS
    spo = (eff_no // BS) if use_oi else 0   # CAP-aware step count -> fixed COCO:OI ratio
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
            if OI_CAP and OI_CAP < NO:
                po = po[:OI_CAP]          # re-drawn fresh each epoch -> still sees whole pool over time
            for s in range(0, po.numel() - BS + 1, BS):
                idx = po[s:s+BS]
                oi_i, oi_t = oi_img[idx].to(dev), oi_txt[idx].to(dev)
                io = img_h(oi_i)
                out = lf(io, txt_h(oi_t), progress=g/max(steps,1))
                total = out["total"] + (W_RKD * rkd_dist(io[-1]["continuous"], oi_i) if use_rkd else 0.0) \
                                     + (W_CROVCA * coding_rate(io[-1]["continuous"]) if use_cr else 0.0)
                opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    with torch.no_grad():
        io_, to_ = img_h(te_i.to(dev)), txt_h(te_t.to(dev))
    q = {"float": eval_rank(te_i @ te_t.T, True),
         f"{MB}bit": eval_rank(hdist(io_[-1]["binary"].cpu().float(), to_[-1]["binary"].cpu().float()), False)}
    print(f"  [{TAG}/{mode}] {time.perf_counter()-t0:.0f}s | COCO float R@10 {q['float']['R@10']} | {MB}bit R@10 {q[f'{MB}bit']['R@10']}", flush=True)
    torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(),
                "bits": BITS, "hidden": P["hidden"], "embed": embed, "mode": mode},
               f"/tmp/mixctrl_head_{TAG}_{mode}.pt")
    return img_h, txt_h, q


heads = {}
for m in MODES:
    print(f"=== [{TAG}] training {m} ===", flush=True)
    heads[m] = train(m)

# ---- FIXED eval corpus + query set (independent of OI_PAIRS) ----
oi = np.ascontiguousarray(np.load(OI_INDEX, allow_pickle=True)["emb"].astype(np.float32))
oin = oi / (np.linalg.norm(oi, axis=1, keepdims=True) + 1e-9)
oit = torch.from_numpy(oin)
def ilog(img_h):
    out = np.empty((len(oin), MB), np.float32)
    with torch.no_grad():
        for i in range(0, len(oin), 8192):
            out[i:i+8192] = img_h(oit[i:i+8192].to(dev))[-1]["continuous"].cpu().numpy()
    return out
logits = {m: ilog(heads[m][0]) for m in MODES}

# FIXED query set from /tmp/oi_pairs.pt (always the same 200 caps, seed 0) — comparable across runs
QSRC = F.normalize(torch.load("/tmp/oi_pairs.pt", map_location="cpu")["txt_emb"], dim=1)
qidx = np.random.default_rng(0).choice(QSRC.shape[0], size=200, replace=False)
qsub = QSRC[qidx]
es = oin @ qsub.numpy().T
emb_top = [set(np.argpartition(-es[:, j], 99)[:100].tolist()) for j in range(qsub.shape[0])]
print(f"\n=== [{TAG}] FIXED-query agreement@100 (200 caps from oi_pairs, seed 0) ===", flush=True)
for m in MODES:
    with torch.no_grad():
        tq = heads[m][1](qsub.to(dev))[-1]["continuous"].cpu().numpy()
    hs = logits[m] @ tq.T
    ag = np.mean([len(set(np.argpartition(-hs[:, j], 99)[:100].tolist()) & emb_top[j]) / 100 for j in range(qsub.shape[0])])
    print(f"  {m:22}: {ag*100:.1f}%  (COCO {MB}bit R@10 {heads[m][2][f'{MB}bit']['R@10']})", flush=True)
print(f"MIXCTRL_DONE_{TAG}", flush=True)
