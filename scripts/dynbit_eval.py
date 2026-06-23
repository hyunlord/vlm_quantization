"""Lever D — dynamic / query-adaptive bit allocation over Matryoshka nested codes.

Eval-only (no training). Reuses a trained nested head (default /tmp/k1024_coco.pt,
bits [8..1024]) to produce COCO 5K test codes, then compares retrieval policies on the
COST-ACCURACY frontier (R@10 vs average bits PROCESSED per gallery comparison):

  uniform      : fixed b-bit Hamming over all gallery (the frontier to beat)
  coarse2fine  : rank all gallery at b_lo, rerank top-K at 1024 (avg bits = b_lo + 1024*K/N)
  cascade      : like coarse2fine but rerank ONLY "hard" queries (small top1-top2 margin
                 at b_lo); easy queries answered at b_lo (query-adaptive avg bits)

Honesty: coarse2fine/cascade reduce retrieval COMPUTE/LATENCY, NOT storage (gallery still
stores 1024-bit codes for rerank). Reports R@{1,5,10}, avg_bits, and wall-clock latency
for EN T2I and KO T2I.

GREEN (gate): match uniform-1024 R@10 at avg_bits <= 60% (=614), OR +1.0pt at matched avg_bits.

Env: HEAD(/tmp/k1024_coco.pt) CSV(/tmp/dynbit.csv)
"""
from __future__ import annotations
import csv, os, sys, time
import torch, torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

HEAD = os.environ.get("HEAD", "/tmp/k1024_coco.pt")
CSV = os.environ.get("CSV", "/tmp/dynbit.csv")
dev = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 5, 10]


def nrm(x):
    return F.normalize(x, dim=1)


ck = torch.load(HEAD, map_location="cpu")
bits, hidden, embed, ni = ck["bits"], ck["hidden"], ck["embed"], ck.get("norm_in", 1)
ih = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval()
th = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval()
ih.load_state_dict(ck["img_h"]); th.load_state_dict(ck["txt_h"])

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
prep = nrm if ni else (lambda x: x)
img = prep(EC["test"]["img"]).to(dev)
en = prep(EC["test"]["txt"]).to(dev)
ko = prep(KO["txt_emb"]).to(dev)
ids = EC["test"]["ids"]
labels = (ids if torch.is_tensor(ids) else torch.tensor([int(x) for x in ids])).to(dev)
N = img.shape[0]

with torch.no_grad():
    IO = ih(img); EO = th(en); KOO = th(ko)
# per-bit binary code banks (gallery=image, queries=text), as float ±1 on device
gal = {b: IO[i]["binary"].float() for i, b in enumerate(bits)}
qry = {"EN": {b: EO[i]["binary"].float() for i, b in enumerate(bits)},
       "KO": {b: KOO[i]["binary"].float() for i, b in enumerate(bits)}}
print(f"head {HEAD} bits {bits} norm_in {ni} | N={N}", flush=True)


def hdist(q, g, b):  # q (Nq,b) g (Ng,b) -> (Nq,Ng)
    return (b - q @ g.t()) / 2


def recall_from_order(order, k_list=KS):
    rel = (labels[order] == labels[:, None])
    return {k: round((rel[:, :k].sum(1) > 0).float().mean().item() * 100, 2) for k in k_list}


def uniform(qb, b):
    d = hdist(qb, gal[b], b)
    return recall_from_order(d.argsort(dim=1))


def coarse2fine(q_lo, q_hi, b_lo, K, b_hi=1024):
    d_lo = hdist(q_lo, gal[b_lo], b_lo)              # (Nq,N)
    short = d_lo.argsort(dim=1)[:, :K]               # (Nq,K) candidate gallery idx
    gh = gal[b_hi]                                   # (N,b_hi)
    cand = gh[short]                                 # (Nq,K,b_hi)
    d_hi = (b_hi - torch.einsum("qkb,qb->qk", cand, q_hi)) / 2  # (Nq,K) rerank
    rerank = short.gather(1, d_hi.argsort(dim=1))    # (Nq,K) reranked gallery idx
    rel = (labels[rerank] == labels[:, None])
    return {k: round((rel[:, :k].sum(1) > 0).float().mean().item() * 100, 2) for k in KS}, b_lo + b_hi * K / N


def cascade(q_lo, q_hi, b_lo, K, frac_thresh, b_hi=1024):
    """rerank only HARD queries (small top1-top2 Hamming margin at b_lo)."""
    d_lo = hdist(q_lo, gal[b_lo], b_lo)
    srt, idx = d_lo.sort(dim=1)
    margin = (srt[:, 1] - srt[:, 0])                 # top1-top2 gap (bits)
    thr = torch.quantile(margin, frac_thresh)         # frac_thresh of queries are "hard"
    hard = margin <= thr
    order = idx.clone()                               # default: low-bit order
    if hard.any():
        short = idx[hard][:, :K]
        cand = gal[b_hi][short]
        d_hi = (b_hi - torch.einsum("qkb,qb->qk", cand, q_hi[hard])) / 2
        rr = short.gather(1, d_hi.argsort(dim=1))
        order[hard, :K] = rr
    rel = (labels[order] == labels[:, None])
    f = hard.float().mean().item()
    avg = b_lo + f * b_hi * K / N
    return {k: round((rel[:, :k].sum(1) > 0).float().mean().item() * 100, 2) for k in KS}, avg, round(f, 3)


def timed(fn, n=20):
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000  # ms/run


rows = []
for lang in ("EN", "KO"):
    Q = qry[lang]
    # uniform frontier
    for b in bits:
        r = uniform(Q[b], b)
        rows.append({"lang": lang, "policy": "uniform", "b_lo": b, "K": 0, "frac_hard": 1.0,
                     "avg_bits": b, **{f"R@{k}": r[k] for k in KS}})
    # coarse-to-fine sweep
    for b_lo in (64, 128, 256):
        for K in (50, 100, 200, 500, 1000):
            r, avg = coarse2fine(Q[b_lo], Q[1024], b_lo, K)
            rows.append({"lang": lang, "policy": "coarse2fine", "b_lo": b_lo, "K": K, "frac_hard": 1.0,
                         "avg_bits": round(avg, 1), **{f"R@{k}": r[k] for k in KS}})
    # cascade (query-adaptive): rerank only hard queries
    for b_lo in (64, 128):
        for frac in (0.2, 0.4, 0.6):
            for K in (200, 500):
                r, avg, f = cascade(Q[b_lo], Q[1024], b_lo, K, frac)
                rows.append({"lang": lang, "policy": "cascade", "b_lo": b_lo, "K": K, "frac_hard": f,
                             "avg_bits": round(avg, 1), **{f"R@{k}": r[k] for k in KS}})

# latency proxy (EN): uniform-1024 vs a representative coarse2fine
lat_uni = timed(lambda: hdist(qry["EN"][1024], gal[1024], 1024).argsort(dim=1))
lat_c2f = timed(lambda: coarse2fine(qry["EN"][64], qry["EN"][1024], 64, 200))
print(f"latency(EN, N={N}): uniform-1024 {lat_uni:.2f}ms | c2f(64,K200) {lat_c2f:.2f}ms | speedup {lat_uni/lat_c2f:.1f}x", flush=True)

w = csv.DictWriter(open(CSV, "w"), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- frontier analysis + gate ----
def best_at_or_below(lang, max_bits):
    cand = [r for r in rows if r["lang"] == lang and r["avg_bits"] <= max_bits]
    return max(cand, key=lambda r: r["R@10"]) if cand else None

print("\n=== COST-ACCURACY FRONTIER (T2I R@10 vs avg_bits) ===")
for lang in ("EN", "KO"):
    uni1024 = next(r for r in rows if r["lang"] == lang and r["policy"] == "uniform" and r["b_lo"] == 1024)
    target = uni1024["R@10"]
    print(f"\n[{lang}] uniform-1024 R@10={target} @1024 bits. uniform frontier:")
    for r in rows:
        if r["lang"] == lang and r["policy"] == "uniform":
            print(f"   uniform {r['b_lo']:>5}b  R@1/5/10 {r['R@1']}/{r['R@5']}/{r['R@10']}")
    # best dynamic at <=614 avg bits (60%)
    dyn = [r for r in rows if r["lang"] == lang and r["policy"] != "uniform" and r["avg_bits"] <= 614]
    dyn.sort(key=lambda r: -r["R@10"])
    print(f"  best dynamic @<=614 avg_bits:")
    for r in dyn[:5]:
        d = r["R@10"] - target
        print(f"   {r['policy']:<11} b_lo{r['b_lo']} K{r['K']} fhard{r['frac_hard']} avg {r['avg_bits']:>6}b  "
              f"R@10 {r['R@10']} (vs uni1024 {d:+.2f})")
    # gate: match target R@10 at <=614, or +1.0 at matched avg_bits
    matched = [r for r in dyn if r["R@10"] >= target - 0.05]
    if matched:
        best = min(matched, key=lambda r: r["avg_bits"])
        print(f"  GATE[{lang}]: matches uni-1024 R@10 ({target}) at {best['avg_bits']}b "
              f"= {100*best['avg_bits']/1024:.0f}% of 1024 ({best['policy']} b_lo{best['b_lo']} K{best['K']}) "
              f"-> {'GREEN' if best['avg_bits'] <= 614 else 'red'}")
    else:
        print(f"  GATE[{lang}]: no dynamic policy matches uni-1024 R@10 at <=614b -> red")
print(f"\nDYNBIT_DONE -> {CSV}")
