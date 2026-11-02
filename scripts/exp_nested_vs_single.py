"""64-bit single-length training vs 64-bit sliced from Matryoshka (nested) training.

Based on scripts/train_1024.py, mode `coco` (no RKD / CroVCA / OI). Differences from train_1024:
  1. 5,000 train images are held out (paper/holdout_idx.npy, fixed seed 42) and never trained on;
     the remaining 108,287 train the heads. The holdout's clean/weak/strong views are the
     near-duplicate evaluation set.
  2. BITS (and, for one auxiliary arm, lcs_weight) is the only thing that varies between arms.
  3. Seed is a parameter (train_1024 hard-coded 42). torch.manual_seed(seed) drives init/dropout;
     the per-epoch data permutation comes from a SEPARATE torch.Generator(seed) so every arm sees
     the identical batch order for a given seed (init draws differ per BITS and would otherwise
     shift the global RNG stream).
  4. Evaluation is at 64 bits only (output of BN_64), plus prefix-consistency diagnostics.

Arms:
  S64          BITS=[64]                         (LCS inactive: single length)
  N1024        BITS=[8,16,32,64,128,256,512,1024]
  N64          BITS=[8,16,32,64]
  N1024-noLCS  BITS as N1024, lcs_weight=0

Usage (DGX):  .venv/bin/python scripts/exp_nested_vs_single.py --arms S64,N1024 --seeds 42,123,7
Writes one JSON per run to paper/nvs_runs/<arm>_s<seed>.json; aggregate with
scripts/exp_nested_vs_single_report.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

ARMS = {
    "S64": {"bits": [64], "lcs": None},
    "N1024": {"bits": [8, 16, 32, 64, 128, 256, 512, 1024], "lcs": None},
    "N64": {"bits": [8, 16, 32, 64], "lcs": None},
    "N1024-noLCS": {"bits": [8, 16, 32, 64, 128, 256, 512, 1024], "lcs": 0.0},
}
EVAL_BIT = 64
EPOCHS = int(os.environ.get("EPOCHS", "25")); BS = 512
NORM_IN = 1
N_HOLDOUT = 5000; HOLDOUT_SEED = 42
N_PAIRS = 1_000_000; PAIR_SEED = 0
RADII = [0, 1, 2, 3]
KS = [1, 5, 10]
# Paper table values (brief) — used only when /tmp/hp_results.json is absent.
PAPER_HP = {"hidden": 384, "dropout": 0.1, "lr": 1e-3, "wd": 0.01, "temperature": 0.07,
            "ortho": 0.1, "quant": 0.1, "balance": 0.01, "cons": 0.5, "lcs": 0.5}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_hp():
    p = Path("/tmp/hp_results.json")
    if p.exists():
        return json.load(open(p))["best_params"], "hp_results.json"
    return dict(PAPER_HP), "brief/paper table (hp_results.json absent)"


def nrm(x):
    return F.normalize(x, dim=1)


def holdout_indices(n_total: int) -> np.ndarray:
    path = Path(REPO) / "paper" / "holdout_idx.npy"
    idx = np.sort(np.random.default_rng(HOLDOUT_SEED).choice(n_total, N_HOLDOUT, replace=False))
    if path.exists():
        saved = np.load(path)
        assert np.array_equal(saved, idx), "paper/holdout_idx.npy differs from the seed-42 draw"
    else:
        path.parent.mkdir(parents=True, exist_ok=True); np.save(path, idx)
    return idx


# ---------------- ranking (same logic as train_1024.eval_rank / hdist, plus mAP@10) ----------------
def hdist(q, db):
    return (q.size(1) - q @ db.t()) / 2


def rank_metrics(scores, q_lab, db_lab, larger: bool) -> dict:
    # stable sort: Hamming ties broken by database index (deterministic)
    order = scores.argsort(dim=1, descending=larger, stable=True)
    rel = (db_lab[order] == q_lab[:, None]).float()
    out = {f"R@{k}": (rel[:, :k].sum(1) > 0).float().mean().item() for k in KS}
    r10 = rel[:, :10]
    prec = r10.cumsum(1) / torch.arange(1, 11, dtype=torch.float32)
    n_rel = (db_lab[None, :] == q_lab[:, None]).sum(1).clamp(max=10).float()
    out["mAP@10"] = ((prec * r10).sum(1) / n_rel.clamp(min=1)).mean().item()
    return out


def ranking(img_codes, txt_codes, ids, larger=False):
    s = hdist(img_codes, txt_codes) if not larger else img_codes @ txt_codes.t()
    i2t = rank_metrics(s, ids, ids, larger)
    t2i = rank_metrics(s.t().contiguous(), ids, ids, larger)
    return {**{f"I2T_{k}": v for k, v in i2t.items()}, **{f"T2I_{k}": v for k, v in t2i.items()}}


# ---------------- near-duplicate exact-match metrics (64-bit image codes) ----------------
def hamming_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a != b).sum(1)


def neardup(c: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> tuple[dict, dict]:
    """c/w/s: (N, 64) bool codes of clean/weak/strong views. Returns (scalar metrics, extras)."""
    N = c.shape[0]
    m = {}
    eq_w, eq_s = (c == w).all(1), (c == s).all(1)
    m["group_consistency"] = (eq_w & eq_s).float().mean().item()
    m["pair_match_weak"] = eq_w.float().mean().item()
    m["pair_match_strong"] = eq_s.float().mean().item()
    g = torch.Generator().manual_seed(PAIR_SEED)
    i = torch.randint(0, N, (N_PAIRS,), generator=g)
    j = torch.randint(0, N - 1, (N_PAIRS,), generator=g); j = j + (j >= i).long()  # j != i
    d_inter = hamming_rows(c[i], c[j]); d_w = hamming_rows(c, w); d_s = hamming_rows(c, s)
    m["inter_collision_rate"] = (d_inter == 0).float().mean().item()
    for r in RADII:
        m[f"r{r}_same_weak"] = (d_w <= r).float().mean().item()
        m[f"r{r}_same_strong"] = (d_s <= r).float().mean().item()
        m[f"r{r}_inter"] = (d_inter <= r).float().mean().item()
    # code purity over the N clean codes
    packed = [bytes(np.packbits(row.numpy()).tobytes()) for row in c]
    from collections import Counter
    cnt = Counter(packed); sizes = sorted(cnt.values(), reverse=True)
    m["n_unique_codes"] = float(len(cnt))
    m["code_purity"] = sum(1 for v in sizes if v == 1) / len(sizes)          # buckets holding 1 image
    m["frac_images_unique_code"] = sum(1 for v in sizes if v == 1) / N        # images alone in their bucket
    m["max_bucket"] = float(sizes[0])
    # bit statistics
    act = c.float().mean(0)                       # P(bit = +1) per bit, clean holdout
    flip = (c != s).float().mean(0)               # clean -> strong flip rate per bit
    m["bit_act_mean"] = act.mean().item(); m["bit_act_absdev_mean"] = (act - 0.5).abs().mean().item()
    m["bit_act_min"] = act.min().item(); m["bit_act_max"] = act.max().item()
    m["flip_strong_mean"] = flip.mean().item(); m["flip_strong_std"] = flip.std().item()
    m["flip_strong_min"] = flip.min().item(); m["flip_strong_max"] = flip.max().item()
    m["flip_strong_p50"] = flip.median().item(); m["flip_strong_p90"] = flip.quantile(0.9).item()
    extras = {"top10_bucket_sizes": sizes[:10],
              "bucket_size_hist": {str(k): v for k, v in sorted(Counter(sizes).items())},
              "bit_act": [round(x, 4) for x in act.tolist()],
              "flip_strong": [round(x, 4) for x in flip.tolist()]}
    return m, extras


def train_and_eval(arm: str, seed: int, D: dict, P: dict, hp_src: str) -> dict:
    spec = ARMS[arm]; BITS = spec["bits"]; lcs_w = P["lcs"] if spec["lcs"] is None else spec["lcs"]
    k64 = BITS.index(EVAL_BIT)
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)  # data order: identical across arms for a given seed
    embed = D["clean"].shape[1]; N = D["clean"].shape[0]
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=lcs_w,
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    spc = N // BS; steps = EPOCHS * spc
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    print(f"[{arm} s{seed}] BITS={BITS} lcs_w={lcs_w} N={N} steps/epoch={spc} OneCycle total_steps={steps}", flush=True)
    clean, weak, strong, txt = (D[k] for k in ("clean", "weak", "strong", "txt"))
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        pc = torch.randperm(N, generator=gen)
        for s in range(0, N - BS + 1, BS):
            idx = pc[s:s + BS].to(dev)
            ci, wi, si, ti = clean[idx], weak[idx], strong[idx], txt[idx]
            io = img_h(ci)
            out = lf(io, txt_h(ti), weak_image_outputs=img_h(wi), aug_image_outputs=img_h(si), progress=g / max(steps, 1))
            opt.zero_grad(); out["total"].backward(); opt.step(); sched.step(); g += 1
        if ep in (0, EPOCHS - 1):
            print(f"  ep{ep+1} loss {out['total'].item():.4f} lcs {out['lcs'].item():.4f}", flush=True)
    train_sec = time.perf_counter() - t0
    assert g == steps, (g, steps)

    img_h.eval(); txt_h.eval()
    res = {"arm": arm, "seed": seed, "bits": BITS, "lcs_weight": lcs_w, "eval_bit": EVAL_BIT,
           "n_train": N, "steps": steps, "train_sec": round(train_sec, 1), "hp": P, "hp_source": hp_src}
    with torch.no_grad():
        ti_o, tt_o = img_h(D["te_i"]), txt_h(D["te_t"])
        hc, hw, hs = img_h(D["ho_c"]), img_h(D["ho_w"]), img_h(D["ho_s"])
    b = lambda o: o[k64]["binary"].float().cpu()
    metrics = ranking(b(ti_o), b(tt_o), D["te_ids"])
    nd, extras = neardup(b(hc) > 0, b(hw) > 0, b(hs) > 0)
    metrics.update(nd)
    res["metrics"] = metrics; res["extras"] = extras
    if 1024 in BITS:  # BN_64 output vs first 64 bits of the BN_1024 output
        k1024 = BITS.index(1024); pre = {}
        for name, o in (("test_img", ti_o), ("test_txt", tt_o), ("holdout_img_clean", hc)):
            a64 = o[k64]["binary"] > 0; a1024 = o[k1024]["binary"][:, :EVAL_BIT] > 0
            pre[name] = {"sample_exact_match": (a64 == a1024).all(1).float().mean().item(),
                         "bit_agreement": (a64 == a1024).float().mean().item(), "n": int(a64.shape[0])}
        res["prefix_consistency"] = pre
    torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(), "bits": BITS, "hidden": P["hidden"],
                "embed": embed, "arm": arm, "seed": seed, "norm_in": NORM_IN}, f"/tmp/nvs_{arm}_s{seed}.pt")
    print(f"  [{arm} s{seed}] {train_sec:.0f}s | 64b I2T R@10 {metrics['I2T_R@10']:.4f} T2I R@10 {metrics['T2I_R@10']:.4f}"
          f" | group_cons {metrics['group_consistency']:.4f} inter_coll {metrics['inter_collision_rate']:.2e}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="S64,N1024")
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--out-dir", default=f"{REPO}/paper/nvs_runs")
    a = ap.parse_args()
    P, hp_src = load_hp()
    TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")["test"]
    n_total = TC["clean"].shape[0]
    hold = holdout_indices(n_total)
    mask = np.ones(n_total, dtype=bool); mask[hold] = False
    tr = torch.from_numpy(np.nonzero(mask)[0]); ho = torch.from_numpy(hold)
    D = {k: nrm(TC[k][tr]).to(dev) for k in ("clean", "weak", "strong", "txt")}
    D.update({"ho_c": nrm(TC["clean"][ho]).to(dev), "ho_w": nrm(TC["weak"][ho]).to(dev), "ho_s": nrm(TC["strong"][ho]).to(dev),
              "te_i": nrm(EC["img"]).to(dev), "te_t": nrm(EC["txt"]).to(dev), "te_ids": EC["ids"].long()})
    print(f"train {len(tr)} holdout {len(ho)} test {EC['img'].shape[0]} | HP from {hp_src}: {P}", flush=True)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fl = ranking(D["te_i"].cpu(), D["te_t"].cpu(), D["te_ids"], larger=True)  # float cosine ceiling
    json.dump({"arm": "float_cosine", "metrics": fl}, open(out_dir / "float_cosine.json", "w"), indent=1)
    print("float cosine ceiling:", {k: round(v, 4) for k, v in fl.items()}, flush=True)
    for arm in a.arms.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            r = train_and_eval(arm, seed, D, P, hp_src)
            json.dump(r, open(out_dir / f"{arm}_s{seed}.json", "w"), indent=1)
    print("NVS_DONE", a.arms, a.seeds, flush=True)


if __name__ == "__main__":
    main()
