"""Head-structure gate: train baseline NestedHashLayer vs multiscale/UNet variants on cached
frozen embeddings, eval PER-SCALE independent Hamming R@1/R@10 (EN + KO), >=2 seeds.

All variants share the SAME data / loss / scales / optimizer — only the head structure differs,
so Delta-vs-baseline is the gate. Each scale's code is evaluated INDEPENDENTLY (its own bits).

Train data: emb_cache[train] EN (8K pairs) + coco_ko_pairs KO (subsampled), L2-normed (norm_in=1).
Eval: COCO-5K test, T2I (text->image, diagonal gold). EN=emb_cache[test], KO=coco_ko_test (aligned).

Outputs paper/head_multiscale.csv + paper/head_multiscale_summary.json. Run on DGX:
  REPO=$(pwd) .venv/bin/python scripts/head_multiscale_train.py
"""
from __future__ import annotations
import csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.multiscale_heads import build_head, DEFAULT_BITS
from src.losses.combined import CombinedHashLoss

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUTp = os.path.join(REPO, "paper"); os.makedirs(OUTp, exist_ok=True)
BITS = DEFAULT_BITS
VARIANTS = ["baseline", "conv", "convunet", "depth", "depth_skip", "depth_topdown", "residual"]
GATE = {"baseline": "baseline", "conv": "1", "convunet": "1",
        "depth": "3", "depth_skip": "3", "depth_topdown": "3", "residual": "2"}
SEEDS = [0, 1]
KO_N = 16000; EPOCHS = 30; BS = 512; LR = 1e-3


def l2(x):
    return F.normalize(x.float(), p=2, dim=1)


def load_data():
    ec = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    tr = ec["train"]; te = ec["test"]
    en_img, en_txt = l2(tr["img"]), l2(tr["txt"])                 # 8K EN
    ko = torch.load("/tmp/coco_ko_pairs.pt", map_location="cpu")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(ko["img_emb"].shape[0], generator=g)[:KO_N]
    ko_img, ko_txt = l2(ko["img_emb"][idx]), l2(ko["txt_emb"][idx])
    img = torch.cat([en_img, ko_img], 0); txt = torch.cat([en_txt, ko_txt], 0)
    # test
    te_img = l2(te["img"]); te_txt_en = l2(te["txt"])
    kote = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_txt_ko = l2(kote["txt_emb"])                               # aligned (same order, verified)
    return (img, txt), (te_img, te_txt_en, te_txt_ko)


def r_at_k(qb, gb, ks=(1, 10)):
    score = qb @ gb.t()                                          # higher = nearer (Hamming-equiv)
    gold = torch.arange(qb.shape[0], device=qb.device)
    out = {}
    for k in ks:
        hit = (score.topk(k, 1).indices == gold[:, None]).any(1)
        out[k] = round(hit.float().mean().item() * 100, 2)
    return out


@torch.no_grad()
def evaluate(img_h, txt_h, te_img, te_txt_en, te_txt_ko):
    img_h.eval(); txt_h.eval()
    oi = img_h(te_img.to(dev)); oe = txt_h(te_txt_en.to(dev)); ok = txt_h(te_txt_ko.to(dev))
    rows = {}
    for i, b in enumerate(BITS):
        gb = oi[i]["binary"]
        en = r_at_k(oe[i]["binary"], gb); ko = r_at_k(ok[i]["binary"], gb)
        rows[b] = {"r1_en": en[1], "r10_en": en[10], "r1_ko": ko[1], "r10_ko": ko[10]}
    return rows


def train_one(variant, seed, data, test):
    torch.manual_seed(seed); np.random.seed(seed)
    img, txt = data; N = img.shape[0]
    img_h = build_head(variant, 1152, 384, BITS, 0.1).to(dev)
    txt_h = build_head(variant, 1152, 384, BITS, 0.1).to(dev)
    # materialize lazy params (conv heads) with a dummy forward
    with torch.no_grad():
        _ = img_h(img[:8].to(dev)); _ = txt_h(txt[:8].to(dev))
    loss_fn = CombinedHashLoss(BITS).to(dev)
    params = list(img_h.parameters()) + list(txt_h.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * ((N + BS - 1) // BS); step = 0
    img_h.train(); txt_h.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        for s in range(0, N, BS):
            bi = perm[s:s + BS]
            xi = img[bi].to(dev); xt = txt[bi].to(dev)
            io = img_h(xi); to = txt_h(xt)
            out = loss_fn(io, to, progress=step / max(1, total_steps))
            opt.zero_grad(); out["total"].backward(); opt.step(); step += 1
    rows = evaluate(img_h, txt_h, *test)
    n_params = sum(p.numel() for p in img_h.parameters())
    # latency: head forward for 1000 samples (eval mode)
    img_h.eval()
    with torch.no_grad():
        xb = test[0][:1000].to(dev)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5): _ = img_h(xb)
        if dev == "cuda": torch.cuda.synchronize()
        lat_ms = round((time.time() - t0) / 5 * 1000, 3)
    return rows, n_params, lat_ms


def main():
    data, test = load_data()
    print(f"train N={data[0].shape[0]} (EN8K+KO{KO_N}) | test {test[0].shape[0]}", flush=True)
    all_rows = []; summary = {}
    for variant in VARIANTS:
        per_seed = []; npar = lat = None
        for seed in SEEDS:
            try:
                rows, npar, lat = train_one(variant, seed, data, test)
            except Exception as e:
                print(f"  [{variant} seed{seed}] FAIL: {e}", flush=True); continue
            per_seed.append(rows)
            for b in BITS:
                r = rows[b]
                all_rows.append({"gate": GATE[variant], "variant": variant, "seed": seed, "scale": b,
                                 "r1_en": r["r1_en"], "r10_en": r["r10_en"],
                                 "r1_ko": r["r1_ko"], "r10_ko": r["r10_ko"],
                                 "head_params": npar, "latency_ms_1k": lat})
            line = " ".join(f"{b}:{rows[b]['r10_en']:.1f}/{rows[b]['r10_ko']:.1f}" for b in BITS)
            print(f"  [{variant} s{seed}] R@10 EN/KO  {line}  | {npar/1e3:.0f}K par {lat}ms", flush=True)
        if per_seed:
            mean = {b: {m: round(float(np.mean([ps[b][m] for ps in per_seed])), 2)
                        for m in ("r1_en", "r10_en", "r1_ko", "r10_ko")} for b in BITS}
            std = {b: round(float(np.std([ps[b]["r10_en"] for ps in per_seed])), 2) for b in BITS}
            summary[variant] = {"gate": GATE[variant], "mean": mean, "r10_en_std": std,
                                "head_params": npar, "latency_ms_1k": lat, "n_seeds": len(per_seed)}
    with open(os.path.join(OUTp, "head_multiscale.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    # deltas vs baseline (mean over seeds)
    base = summary.get("baseline", {}).get("mean", {})
    deltas = {}
    for v, s in summary.items():
        if v == "baseline": continue
        deltas[v] = {b: {m: round(s["mean"][b][m] - base[b][m], 2) for m in ("r10_en", "r10_ko")} for b in BITS}
    json.dump({"summary": summary, "deltas_vs_baseline": deltas, "bits": BITS, "seeds": SEEDS,
               "note": "Per-scale independent Hamming R@{1,10} T2I EN/KO. Variants vs baseline NestedHashLayer, identical data/loss/optimizer. EN8K+KO16K train, COCO-5K test."},
              open(os.path.join(OUTp, "head_multiscale_summary.json"), "w"))
    print("HEAD_MS_DONE -> paper/head_multiscale.csv, paper/head_multiscale_summary.json", flush=True)


if __name__ == "__main__":
    main()
