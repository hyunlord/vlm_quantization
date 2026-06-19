"""(A) Fine-tune an existing best head on Korean — measures catastrophic forgetting.

Contrast with (B) train_1024.py from-scratch (best recipe + Korean in the pool):
here we LOAD a finished best head and *continue* training it on Korean pairs only,
at low lr for few epochs. EN is expected to drop (forgetting); KO to rise. MIX_EN>0
mixes English COCO batches back in to soften the forgetting (0 = Korean-only = worst).

Env: BASE(/tmp/sweep_c226574_coco_oi_rkd_crovca.pt), KO_PAIRS(/tmp/coco_ko_pairs.pt),
     FT_EPOCHS(5), LR_SCALE(0.1), MIX_EN(0.0), SAVE(/tmp/ft_ko.pt)
"""
from __future__ import annotations
import json, os, sys, time
import torch, torch.nn.functional as F

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.environ.get("BASE", "/tmp/sweep_c226574_coco_oi_rkd_crovca.pt")
KO_PAIRS = os.environ.get("KO_PAIRS", "/tmp/coco_ko_pairs.pt")
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "5"))
LR_SCALE = float(os.environ.get("LR_SCALE", "0.1"))
MIX_EN = float(os.environ.get("MIX_EN", "0.0"))
SAVE = os.environ.get("SAVE", "/tmp/ft_ko.pt")
BS = 512
P = json.load(open("/tmp/hp_results.json"))["best_params"]

ck = torch.load(BASE, map_location="cpu")
BITS, H, E, NI = ck["bits"], ck["hidden"], ck["embed"], ck.get("norm_in", 1)


def nrm(x):
    return F.normalize(x, dim=1)


def inp(x):
    return nrm(x) if NI else x


img_h = NestedHashLayer(E, H, BITS, P["dropout"]).to(dev)
txt_h = NestedHashLayer(E, H, BITS, P["dropout"]).to(dev)
img_h.load_state_dict(ck["img_h"]); txt_h.load_state_dict(ck["txt_h"])

KO = torch.load(KO_PAIRS, map_location="cpu")
ki, kt = inp(KO["img_emb"]), inp(KO["txt_emb"]); NK = ki.shape[0]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
ci, ct = inp(TC["clean"]), inp(TC["txt"]); NC = ci.shape[0]

lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                      balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                      temperature=P["temperature"]).to(dev)
opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()),
                        lr=P["lr"] * LR_SCALE, weight_decay=P["wd"])
spe = NK // BS; steps = FT_EPOCHS * spe; g = 0
print(f"FT BASE={BASE.split('/')[-1]} NK={NK:,} MIX_EN={MIX_EN} epochs={FT_EPOCHS} "
      f"lr={P['lr']*LR_SCALE:.2e} norm_in={NI}", flush=True)
t0 = time.perf_counter()
img_h.train(); txt_h.train(); lf.train()
for ep in range(FT_EPOCHS):
    perm = torch.randperm(NK)
    for s in range(0, NK - BS + 1, BS):
        idx = perm[s:s+BS]
        bi, bt = ki[idx].to(dev), kt[idx].to(dev)
        loss = lf(img_h(bi), txt_h(bt), progress=g / max(steps, 1))["total"]
        if MIX_EN > 0 and torch.rand(1).item() < MIX_EN:
            ei = torch.randint(0, NC, (BS,))
            loss = loss + lf(img_h(ci[ei].to(dev)), txt_h(ct[ei].to(dev)), progress=g / max(steps, 1))["total"]
        opt.zero_grad(); loss.backward(); opt.step(); g += 1
img_h.eval(); txt_h.eval()
torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(), "bits": BITS,
            "hidden": H, "embed": E, "mode": "ft_korean", "norm_in": NI, "base": BASE}, SAVE)
print(f"FT_DONE {time.perf_counter()-t0:.0f}s -> {SAVE}", flush=True)
