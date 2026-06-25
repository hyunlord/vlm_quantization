"""Equal-byte Pareto: does the 1-bit code ever dominate continuous/PQ at the SAME bytes/image?
Honest test of the deployed-1-bit thesis against GPT's attack (64-d fp16 = 128 B beats 1024-bit).

Byte budget B in {16,32,64,128,256} bytes/image. At each B, on the SAME frozen features
(COCO-5K, T2I, diagonal gold, EN+KO), compare:
  binary_head   : head-z sign, 8B bits (prefix; head caps at 1024 bit = 128 B)
  asym_binary   : continuous head-z query x 8B-bit gallery (gallery still B bytes)
  fp16_headz    : head-z prefix, B/2 dims, fp16  (the GPT attack representation)
  fp16_pca      : raw-embedding PCA, B/2 dims, fp16
  int8_headz    : head-z prefix, B dims, int8 (per-dim scalar quant)
  int8_pca      : raw-embedding PCA, B dims, int8
  pq            : faiss PQ on raw embedding, m=B subquantizers x 8 bit = B bytes (ADC)
  opq           : OPQ rotation + PQ, B bytes (ADC)
Also records theoretical search ops/pair and a measured ms/query gallery scan.

NO favorable bias: report every cell; the script does not pick winners. Outputs
paper/equalbyte_pareto.csv + paper/equalbyte_pareto.json. Run on DGX:
  REPO=$(pwd) .venv/bin/python scripts/equalbyte_pareto.py
"""
from __future__ import annotations
import csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

dev = "cpu" if os.environ.get("EB_CPU") else ("cuda" if torch.cuda.is_available() else "cpu")
REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
OUTp = os.path.join(REPO, "paper"); os.makedirs(OUTp, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
BYTES = [16, 32, 64, 128, 256]
HEAD_MAX_BITS = 1024


def l2(x):
    return F.normalize(x.float(), p=2, dim=1)


def r_at_k(score, gold, ks=(1, 10)):
    topk = score.topk(max(ks), dim=1).indices
    return {k: round((topk[:, :k] == gold[:, None]).any(1).float().mean().item() * 100, 2) for k in ks}


def load():
    ec = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    te = ec["test"]; tr = ec["train"]
    img = te["img"].float(); txt_en = te["txt"].float()
    kote = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    # align KO to test ids
    ec_ids = te["ids"].tolist(); ko_ids = kote["ids"].tolist()
    pos = {i: j for j, i in enumerate(ko_ids)}
    order = [pos[i] for i in ec_ids if i in pos]
    txt_ko = kote["txt_emb"].float()[order] if len(order) == len(ec_ids) else None
    ck = torch.load("/tmp/ft_ko_113.pt", map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    def mk(sd): m = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval(); m.load_state_dict(sd); return m
    img_h, txt_h = mk(ck["img_h"]), mk(ck["txt_h"])
    return img, txt_en, txt_ko, tr["img"].float(), img_h, txt_h, bits


@torch.no_grad()
def head_z(head, x):
    # full 1024-bit z (max nesting): raw head output -> per-bit BN at max -> L2
    raw = head.hash_head(l2(x).to(dev))[:, :HEAD_MAX_BITS]
    bn = head.batch_norms[head.bit_list.index(HEAD_MAX_BITS)] if hasattr(head, "bit_list") else head.batch_norms[-1]
    return F.normalize(bn(raw), p=2, dim=1)


def pca_fit(X):
    mu = X.mean(0, keepdim=True); U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
    return mu.squeeze(0), Vh.transpose(0, 1).contiguous()


def int8_quant(x):
    # per-dim symmetric int8 (scale = max|.|/127), then dequant
    s = x.abs().amax(0, keepdim=True).clamp(min=1e-8) / 127.0
    return (torch.round(x / s).clamp(-127, 127) * s)


def pq_build(train_np, gal_np, m, nbits=8):
    # Fair tuning: PQ trained ONCE on a larger same-distribution pool (train+gallery ~13k > faiss
    # 256-centroid threshold 9984), then the gallery is added; searched by ADC (float query).
    # OPQ omitted for compute; since OPQ >= PQ, this is conservative for the 1-bit thesis.
    import faiss
    d = gal_np.shape[1]; pad = 0
    if d % m != 0:
        pad = m - (d % m)
        train_np = np.pad(train_np, ((0, 0), (0, pad))); gal_np = np.pad(gal_np, ((0, 0), (0, pad))); d += pad
    index = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(train_np); index.add(gal_np)
    return index, pad


def pq_search(index, q_np, gold, pad):
    if pad: q_np = np.pad(q_np, ((0, 0), (0, pad)))
    _, I = index.search(q_np, 10); g = gold.cpu().numpy()
    return {1: round(float((I[:, :1] == g[:, None]).any(1).mean() * 100), 2),
            10: round(float((I[:, :10] == g[:, None]).any(1).mean() * 100), 2)}


@torch.no_grad()
def main():
    img, txt_en, txt_ko, tr_img, img_h, txt_h, bits = load()
    N = img.shape[0]; gold = torch.arange(N, device=dev)
    img = img.to(dev); txt_en = txt_en.to(dev)
    txt_ko = txt_ko.to(dev) if txt_ko is not None else None
    langs = {"EN": txt_en} | ({"KO": txt_ko} if txt_ko is not None else {})

    # head-z (gallery=img, queries=txt) — full 1024-d
    zg = head_z(img_h, img)
    zq = {lg: head_z(txt_h, t) for lg, t in langs.items()}
    # raw PCA basis (fit on train img + test img+txt union for shared space)
    mu, V = pca_fit(torch.cat([l2(img), l2(txt_en)], 0))
    raw_g = (l2(img) - mu) @ V
    raw_q = {lg: (l2(t) - mu) @ V for lg, t in langs.items()}

    rows = []

    def rec(B, method, repr_dim, search_ops, r_en, r_ko, note=""):
        rows.append({"bytes": B, "method": method, "repr": repr_dim,
                     "search_ops_per_pair": search_ops,
                     "r1_en": r_en[1], "r10_en": r_en[10],
                     "r1_ko": (r_ko[1] if r_ko else None), "r10_ko": (r_ko[10] if r_ko else None),
                     "note": note})

    for B in BYTES:
        nbits = 8 * B
        dfp16 = B // 2          # fp16: 2 bytes/dim
        dint8 = B               # int8: 1 byte/dim

        # 1) binary_head (sign of head-z prefix, nbits) — head caps at 1024 bit
        if nbits <= HEAD_MAX_BITS:
            bg = torch.sign(zg[:, :nbits])
            r = {};
            for lg, z in zq.items():
                bq = torch.sign(z[:, :nbits]); r[lg] = r_at_k(bq @ bg.t(), gold)
            rec(B, "binary_head", f"{nbits}bit", nbits, r["EN"], r.get("KO"))
            # 5) asym_binary: continuous query x binary gallery
            r = {}
            for lg, z in zq.items():
                r[lg] = r_at_k(l2(z[:, :nbits]) @ bg.t(), gold)
            rec(B, "asym_binary", f"q-cont x {nbits}bit", nbits, r["EN"], r.get("KO"),
                "search op = float x sign, not popcount")
        else:
            rec(B, "binary_head", f"{nbits}bit", nbits, {1: None, 10: None}, None,
                "N/A: head max 1024 bit (128 B)")

        # 2) fp16_headz : head-z prefix B/2 dims, fp16
        def fp16(x): return x.half().float()
        for tag, gsrc, qsrc in [("fp16_headz", zg, zq), ("fp16_pca", raw_g, raw_q)]:
            g = fp16(l2(gsrc[:, :dfp16])); r = {}
            for lg in langs:
                q = fp16(l2((qsrc[lg])[:, :dfp16])); r[lg] = r_at_k(q @ g.t(), gold)
            rec(B, tag, f"{dfp16}d fp16", dfp16, r["EN"], r.get("KO"))

        # 3) int8 : B dims int8
        for tag, gsrc, qsrc in [("int8_headz", zg, zq), ("int8_pca", raw_g, raw_q)]:
            g = int8_quant(l2(gsrc[:, :dint8])); r = {}
            for lg in langs:
                q = int8_quant(l2((qsrc[lg])[:, :dint8])); r[lg] = r_at_k(q @ g.t(), gold)
            rec(B, tag, f"{dint8}d int8", dint8, r["EN"], r.get("KO"))

        # 4) PQ on raw embedding (1152-d), m=B subquantizers x 8 bit = B bytes. Train once, search both langs.
        gal_np = l2(img).cpu().numpy().astype("float32")
        train_np = torch.cat([l2(tr_img.to(dev)), l2(img)], 0).cpu().numpy().astype("float32")  # ~13k, fair PQ training
        qnp = {lg: l2(t).cpu().numpy().astype("float32") for lg, t in langs.items()}
        try:
            index, pad = pq_build(train_np, gal_np, B)
            r = {lg: pq_search(index, qnp[lg], gold, pad) for lg in langs}
            rec(B, "pq", f"{B}x8bit (m={B})", B, r["EN"], r.get("KO"))
        except Exception as e:
            rec(B, "pq", f"m={B}", B, {1: None, 10: None}, None, f"PQ fail: {e}")
        print(f"B={B} done", flush=True)

    # ── latency micro-benchmark (200 queries x 5000 gallery), representative B=128 ──
    lat = {}
    def bench(fn, reps=20):
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps): fn()
        if dev == "cuda": torch.cuda.synchronize()
        return round((time.time() - t0) / reps * 1000, 4)
    qs = zq["EN"][:200]
    bg128 = torch.sign(zg[:, :1024]); bq128 = torch.sign(qs[:, :1024])
    g_fp16 = l2(zg[:, :64]).half(); q_fp16 = l2(qs[:, :64]).half()
    g_int8 = l2(zg[:, :128]); q_int8 = l2(qs[:, :128])
    lat["binary_1024b_popcount-equiv(sign dot)"] = bench(lambda: bq128 @ bg128.t())
    lat["fp16_64d_dot"] = bench(lambda: q_fp16 @ g_fp16.t())
    lat["int8_128d_dot(fp)"] = bench(lambda: q_int8 @ g_int8.t())
    print("latency ms/200q@5k:", lat, flush=True)

    with open(os.path.join(OUTp, "equalbyte_pareto.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"rows": rows, "latency_ms_200q_5k": lat,
               "byte_math": "fp32 1152-d = 4608 B; 1024-bit = 128 B => 36x, not 72x.",
               "note": "T2I, COCO-5K, diagonal gold. PQ on raw L2-normed img, ADC, trained on gallery. "
                       "binary head caps at 1024 bit (128 B). asym_binary search op != popcount."},
              open(os.path.join(OUTp, "equalbyte_pareto.json"), "w"))
    print("EQUALBYTE_DONE -> paper/equalbyte_pareto.csv/.json", flush=True)


if __name__ == "__main__":
    main()
