"""LOSS-COMPOSITION GATE v2 — CODE-DEFINING regime (AAAI).

v1 (paper-loss-composition, f0af984) measured W/C/U in the ANCHOR-ADAPTATION regime
(target b = sign(frozen anchor), already binary) and found them INERT — the frozen
anchor + BatchNorm fix the geometry, so loss composition cannot matter. NEGATIVE, kept.

v2 measures the regime where composition CAN matter: NO anchor. Both img_h and txt_h
are trained FROM SCRATCH on frozen SigLIP2-So400m features, jointly DEFINING the
1024-bit code space (the original ft113 training setup). Here quant/margin/annealing
shape the code geometry, so composition can drive performance.

Ingredients (identical to v1; symmetric over the two trainable heads):
    align   = InfoNCE(tanh(z_txt/T), tanh(z_img/T))      [pull paired codes together]
    quant   = EAQL(tanh(z_txt/T)) + EAQL(tanh(z_img/T))  [push both away from 0]
    margin  = relu(m - |z_txt|) + relu(m - |z_img|)      [both pre-signs clear of 0 by m]
    nesting = LCS(tanh(z_txt)) + LCS(tanh(z_img))        [Matryoshka long->short]
Compositions:
  (W) weighted-sum : L = align + wq*quant + wm*margin + wl*nesting, T=1 fixed.
  (C) curriculum   : T annealed T0->1 (cosine); wq,wm ramp 0->target over first 60%.
  (U) unified      : U = relu(m - |z_txt|)^2 + relu(m - |z_img|)^2; L = align + wu*U + wl*nesting.

Data (cached SigLIP2 features, frozen encoder):
  train = /tmp/emb_aug.pt  (clean img 113287x1152, txt 113287x1152, ids)
  test  = /tmp/emb_cache.pt (test img/txt 5000x1152 [txt = SigLIP2-text EN], ids, captions)
Eval: gallery = sign(img_h(test_img)); query = sign(txt_h(test_txt)); faiss IndexBinaryFlat
  at nested prefix bits {64,128,256,512,1024}; R@{1,5,10} EN. (KO deferred to a positive gate:
  it needs fresh SigLIP2-text-KO encoding; EN alone decides whether composition matters.)
Diagnostics (1024-bit): sign-margin |z|, noise flip%, paired cosine(z_txt, z_img).

Run on DGX:
  .venv/bin/python web/loss_composition_v2.py --sweep --epochs 8 --out paper/loss_composition_v2.csv
Smoke:
  .venv/bin/python web/loss_composition_v2.py --comp W --epochs 1 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.losses.contrastive import CrossModalContrastiveLoss  # noqa: E402
from src.losses.eaql import EAQLLoss  # noqa: E402
from src.losses.lcs import LCSSelfDistillationLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")       # architecture config only
HP_PATH = os.environ.get("HP_PATH", "/tmp/hp_results.json")
TRAIN_CACHE = os.environ.get("TRAIN_CACHE", "/tmp/emb_aug.pt")
TEST_CACHE = os.environ.get("TEST_CACHE", "/tmp/emb_cache.pt")
EVAL_BITS = [64, 128, 256, 512, 1024]
dev = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
#  Replicate NestedHashLayer.forward but EXPOSE pre-sign z_k (mirrors v1 exactly).
# --------------------------------------------------------------------------- #
def head_z(head: NestedHashLayer, x: torch.Tensor, only_bits=None):
    raw = head.hash_head(x)
    zs = []
    for length, bn in zip(head.bit_list, head.batch_norms):
        sliced = raw[:, :length]
        z = F.normalize(bn(sliced), p=2, dim=1)
        if only_bits is None or length in only_bits:
            zs.append(z)
    return zs


def pack_bits(codes: np.ndarray) -> np.ndarray:
    bits01 = (np.asarray(codes) > 0).astype(np.uint8)
    if bits01.ndim == 1:
        bits01 = bits01[None, :]
    return np.ascontiguousarray(np.packbits(bits01, axis=1, bitorder="big"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
#  Data — frozen SigLIP2 features (no encoder, no e5, no anchor)
# --------------------------------------------------------------------------- #
def build_train_data():
    """Return (img (N,1152) L2, txt (N,1152) L2) paired SigLIP2 train features."""
    d = torch.load(TRAIN_CACHE, map_location="cpu")["train"]
    img = F.normalize(d["clean"].float(), dim=1)
    txt = F.normalize(d["txt"].float(), dim=1)
    assert img.shape[0] == txt.shape[0], (img.shape, txt.shape)
    return img, txt


def build_test_data():
    """Return (img (5000,1152) L2, txt_en (5000,1152) L2, te_ids list)."""
    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    img = F.normalize(EC["img"].float(), dim=1)
    txt = F.normalize(EC["txt"].float(), dim=1)
    te_ids = [int(x) for x in EC["ids"].tolist()]
    return img, txt, te_ids


# --------------------------------------------------------------------------- #
#  Loss — SYMMETRIC, both heads trainable (code-defining)
# --------------------------------------------------------------------------- #
def margin_value(z_txt, z_img, rel):
    with torch.no_grad():
        s = torch.stack([z.std() for z in z_txt] + [z.std() for z in z_img]).mean()
    return rel * float(s)


def compute_loss(comp, zt, zi, contrastive, eaql, lcs, T, wq, wm, wl, wu, m_abs):
    """zt/zi: lists of per-bit pre-sign for txt/img heads (both trainable)."""
    align = torch.zeros((), device=zt[0].device)
    quant = torch.zeros((), device=zt[0].device)
    margin = torch.zeros((), device=zt[0].device)
    nb = len(zt)
    tzt = [torch.tanh(z / T) for z in zt]
    tzi = [torch.tanh(z / T) for z in zi]
    for k in range(nb):
        align = align + contrastive(tzt[k], tzi[k])
        quant = quant + eaql(tzt[k]) + eaql(tzi[k])
        if comp == "U":
            margin = margin + (F.relu(m_abs - zt[k].abs()) ** 2).mean() \
                            + (F.relu(m_abs - zi[k].abs()) ** 2).mean()
        else:
            margin = margin + F.relu(m_abs - zt[k].abs()).mean() \
                            + F.relu(m_abs - zi[k].abs()).mean()
    align = align / nb
    quant = quant / nb
    margin = margin / nb
    nesting = lcs(tzt) + lcs(tzi)
    if comp == "U":
        return align + wu * margin + wl * nesting
    return align + wq * quant + wm * margin + wl * nesting


# --------------------------------------------------------------------------- #
#  Train one configuration — BOTH heads from scratch, joint optimizer
# --------------------------------------------------------------------------- #
def train_one(comp, setting, hp, img_tr, txt_tr, bits, hidden, embed,
              epochs=8, bs=512, seed=42,
              wq=None, wm=None, wl=None, wu=None, m_rel=0.1, T0=4.0, ramp_frac=0.6):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = img_tr.shape[0]
    img_h = NestedHashLayer(embed, hidden, bits, hp["dropout"]).to(dev).train()
    txt_h = NestedHashLayer(embed, hidden, bits, hp["dropout"]).to(dev).train()

    contrastive = CrossModalContrastiveLoss(hp["temperature"]).to(dev)
    eaql = EAQLLoss().to(dev)
    lcs = LCSSelfDistillationLoss().to(dev)

    params = list(img_h.parameters()) + list(txt_h.parameters())
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["wd"])
    spe = N // bs
    steps = epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"],
                                                total_steps=steps, pct_start=0.3)
    wq = hp["quant"] if wq is None else wq
    wl = hp["lcs"] if wl is None else wl
    wm = 0.1 if wm is None else wm
    wu = 0.2 if wu is None else wu

    g = 0
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N - bs + 1, bs):
            idx = perm[s:s + bs]
            zi = head_z(img_h, img_tr[idx].to(dev))
            zt = head_z(txt_h, txt_tr[idx].to(dev))
            prog = g / max(steps, 1)
            if comp == "C":
                T = 1.0 + (T0 - 1.0) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
                ramp = min(1.0, prog / ramp_frac)
                cur_wq, cur_wm, cur_wu = wq * ramp, wm * ramp, wu * ramp
            else:
                T = 1.0
                cur_wq, cur_wm, cur_wu = wq, wm, wu
            m_abs = margin_value(zt, zi, m_rel)
            loss = compute_loss(comp, zt, zi, contrastive, eaql, lcs,
                                T, cur_wq, cur_wm, wl, cur_wu, m_abs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            g += 1
    img_h.eval(); txt_h.eval()
    return img_h, txt_h


# --------------------------------------------------------------------------- #
#  Eval
# --------------------------------------------------------------------------- #
def _faiss(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits)
    ix.add(packed)
    return ix


def _recall(gal_ix, qcodes, te_ids, k):
    _, I = gal_ix.search(qcodes, k)
    return round(100 * np.mean([te_ids[q] in [te_ids[j] for j in I[q]]
                                for q in range(len(te_ids))]), 2)


def eval_run(img_h, txt_h, te_img, te_txt, te_ids):
    """gallery = img_h codes; query = txt_h codes (EN). Per-bit R@{1,5,10}."""
    with torch.no_grad():
        gal_full = (head_z(img_h, te_img.to(dev), only_bits={1024})[0] > 0).cpu().numpy().astype(np.uint8)
        en_full = (head_z(txt_h, te_txt.to(dev), only_bits={1024})[0] > 0).cpu().numpy().astype(np.uint8)
    res = {}
    for b in EVAL_BITS:
        ix = _faiss(pack_bits(gal_full[:, :b]), b)
        q = pack_bits(en_full[:, :b])
        row = {}
        for k in (1, 5, 10):
            row[f"R{k}_EN"] = _recall(ix, q, te_ids, k)
        res[b] = row
    return res


def diagnostics(img_h, txt_h, te_img, te_txt):
    """1024-bit: sign-margin |z| (txt), noise flip% (txt), paired cosine(z_txt, z_img)."""
    with torch.no_grad():
        zt = head_z(txt_h, te_txt.to(dev), only_bits={1024})[0]
        zi = head_z(img_h, te_img.to(dev), only_bits={1024})[0]
    az = zt.abs()
    cos_pair = float(F.cosine_similarity(zt, zi, dim=1).mean())
    x = te_txt.to(dev)
    sigma = 0.05 * x.norm(dim=1, keepdim=True) / math.sqrt(x.shape[1])
    clean_sign = torch.sign(zt)
    flips = []
    gen = torch.Generator(device=dev).manual_seed(123)
    with torch.no_grad():
        for _ in range(5):
            noise = torch.randn(x.shape, generator=gen, device=dev) * sigma
            zn = head_z(txt_h, F.normalize(x + noise, dim=1), only_bits={1024})[0]
            flips.append(float((torch.sign(zn) != clean_sign).float().mean()))
    return {
        "margin_mean": round(float(az.mean()), 4),
        "margin_p10": round(float(torch.quantile(az.flatten().float(), 0.10)), 4),
        "flip_pct": round(100 * float(np.mean(flips)), 3),
        "cos_pair": round(cos_pair, 4),
    }


# --------------------------------------------------------------------------- #
#  Grid — C and U FIRST (gate-critical), then the W sensitivity grid.
# --------------------------------------------------------------------------- #
def build_grid(hp):
    runs = []
    for T0 in (2.0, 4.0):
        runs.append(("C", f"C_T0{T0:g}", {"T0": T0}, [42]))
    for wu in (0.1, 0.2, 0.4):
        for m_rel in (0.05, 0.1):
            runs.append(("U", f"U_wu{wu:g}_m{m_rel:g}", {"wu": wu, "m_rel": m_rel}, [42]))
    wm_pts = [0.0, 0.1, 0.4]
    wq_pts = [0.06, hp["quant"], 0.25]
    for wq in wq_pts:
        for wm in wm_pts:
            runs.append(("W", f"W_wq{wq:g}_wm{wm:g}", {"wq": wq, "wm": wm}, [42]))
    return runs


CSV_COLS = ["comp", "setting", "seed", "bit", "R1_EN", "R5_EN", "R10_EN",
            "margin_mean", "margin_p10", "flip_pct", "cos_pair"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comp", choices=["W", "C", "U"], default="W")
    p.add_argument("--setting", default="single")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--out", default="/tmp/loss_comp_v2.csv")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wq", type=float, default=None)
    p.add_argument("--wm", type=float, default=None)
    p.add_argument("--wl", type=float, default=None)
    p.add_argument("--wu", type=float, default=None)
    p.add_argument("--m_rel", type=float, default=0.1)
    p.add_argument("--T0", type=float, default=4.0)
    args = p.parse_args()

    t0 = time.perf_counter()
    hp = json.load(open(HP_PATH))["best_params"]
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    hidden, embed = ck["hidden"], ck["embed"]

    img_tr, txt_tr = build_train_data()
    if args.smoke:
        img_tr, txt_tr = img_tr[:8192], txt_tr[:8192]
    te_img, te_txt, te_ids = build_test_data()
    print(f"[setup] train N={img_tr.shape[0]:,} test={te_img.shape[0]} dim={embed} "
          f"bits={bits} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)

    rows = []

    def run(comp, sid, kw, seed, epochs):
        rt = time.perf_counter()
        img_h, txt_h = train_one(comp, sid, hp, img_tr, txt_tr, bits, hidden, embed,
                                 epochs=epochs, bs=args.bs, seed=seed,
                                 wq=kw.get("wq"), wm=kw.get("wm"), wl=kw.get("wl"),
                                 wu=kw.get("wu"), m_rel=kw.get("m_rel", 0.1),
                                 T0=kw.get("T0", 4.0))
        res = eval_run(img_h, txt_h, te_img, te_txt, te_ids)
        diag = diagnostics(img_h, txt_h, te_img, te_txt)
        for b in EVAL_BITS:
            r = res[b]
            rows.append({"comp": comp, "setting": sid, "seed": seed, "bit": b,
                         **{k: r[k] for k in ("R1_EN", "R5_EN", "R10_EN")},
                         "margin_mean": diag["margin_mean"], "margin_p10": diag["margin_p10"],
                         "flip_pct": diag["flip_pct"], "cos_pair": diag["cos_pair"]})
        import csv as _csv
        with open(args.out, "w", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=CSV_COLS); _w.writeheader(); _w.writerows(rows)
        r1024 = res[1024]
        print(f"[run] {comp} {sid} s{seed}: 1024 R@10 EN {r1024['R10_EN']} "
              f"| flip {diag['flip_pct']}% margin {diag['margin_mean']} cos {diag['cos_pair']} "
              f"({(time.perf_counter()-rt)/60:.1f}min)", flush=True)
        return r1024["R10_EN"]

    if args.smoke:
        run(args.comp, "smoke", {}, args.seed, args.epochs)
        print("[smoke] OK", flush=True)
        return

    if not args.sweep:
        kw = {"wq": args.wq, "wm": args.wm, "wl": args.wl, "wu": args.wu,
              "m_rel": args.m_rel, "T0": args.T0}
        run(args.comp, args.setting, kw, args.seed, args.epochs)
    else:
        grid = build_grid(hp)
        best = {"W": (-1, None), "C": (-1, None), "U": (-1, None)}
        for comp, sid, kw, seeds in grid:
            for seed in seeds:
                en = run(comp, sid, kw, seed, args.epochs)
                if en > best[comp][0]:
                    best[comp] = (en, (sid, kw))
        for comp in ("W", "C", "U"):
            sid, kw = best[comp][1]
            run(comp, sid, kw, 0, args.epochs)   # 2nd seed for noise band
        print("[sweep] best: " + json.dumps({c: best[c][1][0] for c in best}), flush=True)

    print(f"[done] wrote {len(rows)} rows -> {args.out} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
