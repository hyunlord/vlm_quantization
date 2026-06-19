"""Unified 1024-bit trainer — produces every comparison option at the SAME
Matryoshka range [8..1024] so all settings are comparable at the best bit.

Merges the three experiment scripts' knobs:
  - broaden modes (coco / coco_crovca / coco_oi_rkd_crovca) + RKD + CroVCA
  - OI_CAP (mixctrl mixture-ratio control: cap OI loop per epoch)
  - NORM_IN (ablate: L2-normalize hash-head input or feed raw)
  - SAVE_PREFIX (distinct output filename per run, no clobber)

Frozen backbone, cached embeddings, hidden from hp_results (384). Light eval
only (COCO max-bit R@10); authoritative metrics come from eval_all.py afterward.

Env: BITS, EPOCHS(25), W_RKD(1.0), W_CROVCA(0.1), OI_PAIRS, OI_CAP(0), NORM_IN(1),
     MODES, SAVE_PREFIX.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256,512,1024").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "25")); BS = 512
W_RKD = float(os.environ.get("W_RKD", "1.0")); W_CROVCA = float(os.environ.get("W_CROVCA", "0.1"))
OI_PAIRS = os.environ.get("OI_PAIRS", "/tmp/cc12m_pairs.pt")
OI_CAP = int(os.environ.get("OI_CAP", "0"))
NORM_IN = int(os.environ.get("NORM_IN", "1"))
MODES = os.environ.get("MODES", "coco,coco_crovca,coco_oi_rkd_crovca").split(",")
SAVE_PREFIX = os.environ.get("SAVE_PREFIX", "k1024_")
dev = "cuda" if torch.cuda.is_available() else "cpu"
MB = BITS[-1]; KS = [1, 5, 10]

P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
def nrm(x): return F.normalize(x, dim=1)
def inp(x): return nrm(x) if NORM_IN else x
clean, weak, strong, txt = inp(TC["clean"]), inp(TC["weak"]), inp(TC["strong"]), inp(TC["txt"])
te_i, te_t, labels = inp(EC["test"]["img"]), inp(EC["test"]["txt"]), EC["test"]["ids"]
embed, N = clean.shape[1], clean.shape[0]
NO = 0
if any("oi" in m for m in MODES):
    OI = torch.load(OI_PAIRS, map_location="cpu"); oi_img, oi_txt = inp(OI["img_emb"]), inp(OI["txt_emb"]); NO = oi_img.shape[0]
eff_no = (min(OI_CAP, NO) if OI_CAP else NO)
print(f"[{SAVE_PREFIX}] NORM_IN={NORM_IN} OI_PAIRS={OI_PAIRS if NO else '-'} NO={NO} OI_CAP={OI_CAP}->eff{eff_no} | N={N} BITS={BITS} MODES={MODES}", flush=True)


def eval_rank(scores, larger):
    order = scores.argsort(dim=1, descending=larger); rel = (labels[order] == labels[:, None]).float()
    return {f"R@{k}": round((rel[:, :k].sum(1) > 0).float().mean().item(), 4) for k in KS}
def hdist(q, db): return (q.size(1) - q @ db.t()) / 2
def rkd_dist(sc, te):
    s = F.normalize(sc, dim=1)
    with torch.no_grad():
        t = F.normalize(te, dim=1); td = torch.cdist(t, t); td = td / (td[td > 0].mean() + 1e-8)
    sd = torch.cdist(s, s); sd = sd / (sd[sd > 0].mean() + 1e-8); return F.huber_loss(sd, td)
def coding_rate(z, eps=0.5):
    z = F.normalize(z, dim=1); B, d = z.shape; cov = z.t() @ z
    R = torch.logdet(torch.eye(d, device=z.device, dtype=z.dtype) + (d / (B * eps)) * cov)
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
    spc = N // BS; spo = (eff_no // BS) if use_oi else 0
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
            total = out["total"] + (W_RKD*rkd_dist(io[-1]["continuous"], ci) if use_rkd else 0.0) \
                                 + (W_CROVCA*coding_rate(io[-1]["continuous"]) if use_cr else 0.0)
            opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1
        if use_oi:
            po = torch.randperm(NO)
            if OI_CAP and OI_CAP < NO: po = po[:OI_CAP]
            for s in range(0, po.numel() - BS + 1, BS):
                idx = po[s:s+BS]; oi_i, oi_t = oi_img[idx].to(dev), oi_txt[idx].to(dev)
                io = img_h(oi_i)
                out = lf(io, txt_h(oi_t), progress=g/max(steps,1))
                total = out["total"] + (W_RKD*rkd_dist(io[-1]["continuous"], oi_i) if use_rkd else 0.0) \
                                     + (W_CROVCA*coding_rate(io[-1]["continuous"]) if use_cr else 0.0)
                opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    with torch.no_grad():
        io_, to_ = img_h(te_i.to(dev)), txt_h(te_t.to(dev))
    r10 = eval_rank(hdist(io_[-1]["binary"].cpu().float(), to_[-1]["binary"].cpu().float()), False)["R@10"]
    out_path = f"/tmp/{SAVE_PREFIX}{mode}.pt"
    torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(),
                "bits": BITS, "hidden": P["hidden"], "embed": embed, "mode": mode, "norm_in": NORM_IN}, out_path)
    print(f"  [{SAVE_PREFIX}{mode}] {time.perf_counter()-t0:.0f}s | {MB}bit COCO R@10 {r10} -> {out_path}", flush=True)


for m in MODES:
    print(f"=== train {SAVE_PREFIX}{m} ===", flush=True); train(m)
print(f"TRAIN1024_DONE {SAVE_PREFIX}", flush=True)
