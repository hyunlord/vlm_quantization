"""Gap B — "retrieval information concentrates in few continuous dims; low-bit loss
comes from binarization (sign), not from dimensionality."

Probes RAW frozen VLM embeddings (no learned head) so the phenomenon can be tested
across multiple frozen backbones. Cross-modal COCO-5K (diagonal gold) + XM3600 36-lang.

Backbones (raw test embeddings cached on DGX /tmp):
  so400m       : /tmp/emb_cache.pt[test]          img/txt 1152-d   (+ KO via coco_ko_test.pt)
  siglip2-base : /tmp/bb_siglip2-base_test.pt      img/en/ko 768-d
  altclip-m18  : /tmp/bb_altclip-m18_test.pt        img/en/ko 1024-d

For each backbone × direction (T2I, I2T) × lang:
  1) PCA dim-sweep d in {2..D}: R@10 for continuous (cosine), sign (Hamming), asymmetric (z_q·b_g)
  2) effective-dim: singular-value spectrum, participation ratio, dims for 90/95/99% var
  3) per-PC retrieval contribution: single-PC R@10; variance-order vs retrieval-order
     cumulative R@10  -> is the retrieval-important order the same as the variance order?
  4) bits×dims grid: d dims at {1-bit sign, 2-bit, float} -> R@10 (equal-total-bit contours)
  5) XM3600 per-language saturation point (so400m)

Outputs paper/gapB_*.csv + viz/data/gapB_*.json. Run on DGX:
  REPO=$(pwd) .venv/bin/python scripts/gapB_analysis.py
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np, torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization")
OUTp = os.path.join(REPO, "paper"); OUTv = os.path.join(REPO, "viz", "data")
os.makedirs(OUTp, exist_ok=True); os.makedirs(OUTv, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

DIMS_FULL = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def l2(x):
    return torch.nn.functional.normalize(x.float(), p=2, dim=1)


def r_at_k(score, gold, k=10):
    """score: (Nq, Ng) higher=nearer. gold: (Nq,) gallery index of the true match."""
    topk = score.topk(k, dim=1).indices            # (Nq, k)
    hit = (topk == gold[:, None]).any(dim=1)
    return round(hit.float().mean().item() * 100, 3)


def pca_fit(combined):
    """combined: (M, D) on dev. Returns mean (D,), V (D,D) cols=components, sv (D,)."""
    mu = combined.mean(0, keepdim=True)
    Xc = combined - mu
    # economy SVD on the data matrix; columns of Vh.T are principal directions
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.transpose(0, 1).contiguous()             # (D, D)
    return mu.squeeze(0), V, S                       # S = singular values (desc)


def project(x, mu, V, d):
    return (x - mu) @ V[:, :d]                        # (N, d) PCA scores


def two_bit(scores):
    """Per-dim 2-bit uniform quantization to 4 signed levels {-1.5,-0.5,0.5,1.5}.
    Threshold at per-dim quantiles (0,.5,1 of |.| dist via std). Returns float codes."""
    s = scores
    sd = s.std(0, keepdim=True).clamp(min=1e-6)
    z = s / sd
    # 4 levels split at -0.674, 0, +0.674 (quartiles of N(0,1)) -> centroids
    lvl = torch.zeros_like(z)
    lvl = torch.where(z < -0.674, torch.full_like(z, -1.5), lvl)
    lvl = torch.where((z >= -0.674) & (z < 0.0), torch.full_like(z, -0.5), lvl)
    lvl = torch.where((z >= 0.0) & (z < 0.674), torch.full_like(z, 0.5), lvl)
    lvl = torch.where(z >= 0.674, torch.full_like(z, 1.5), lvl)
    return lvl


def codes_for(qp, gp, mode):
    """Return (query_repr, gallery_repr) for retrieval given PCA scores qp, gp and a coding mode."""
    if mode == "continuous":
        return l2(qp), l2(gp)
    if mode == "sign":
        return torch.sign(qp), torch.sign(gp)
    if mode == "sign_mc":                            # per-modality mean-centered sign (removes modality gap)
        return torch.sign(qp - qp.mean(0, keepdim=True)), torch.sign(gp - gp.mean(0, keepdim=True))
    if mode == "asym":                               # continuous query, binary gallery
        return l2(qp), torch.sign(gp)
    if mode == "twobit":
        return two_bit(qp), two_bit(gp)
    raise ValueError(mode)


# ───────────────────────── data loaders ─────────────────────────
def load_backbones():
    bbs = {}
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")["test"]
    # so400m EN only — its COCO-KO (coco_ko_test) is the non-lowercased text tower which
    # hub-collapses (degenerate); the proper multilingual cut for so400m is XM3600 (lowercased).
    so = {"img": EC["img"].float(), "en": EC["txt"].float(), "D": EC["img"].shape[1],
          "ids": EC["ids"]}
    bbs["so400m"] = so
    for name, path in [("siglip2-base", "/tmp/bb_siglip2-base_test.pt"),
                       ("altclip-m18", "/tmp/bb_altclip-m18_test.pt")]:
        try:
            d = torch.load(path, map_location="cpu")
            bbs[name] = {"img": d["img"].float(), "en": d["en"].float(),
                         "ko": d["ko"].float(), "D": d["img"].shape[1]}
        except Exception as e:
            print(f"  {name} skip:", e)
    return bbs


# ───────────────────────── main sweeps ─────────────────────────
def sweep_backbone(name, bb):
    """Dim-sweep + bits×dims for one backbone. Returns (dim_rows, bd_rows, recall_payload, bd_payload)."""
    D = bb["D"]
    dims = [d for d in DIMS_FULL if d <= D] + ([D] if D not in DIMS_FULL else [])
    dims = sorted(set(dims))
    img = bb["img"].to(dev)
    texts = {k: bb[k].to(dev) for k in ("en", "ko") if k in bb}
    dim_rows, bd_rows = [], []
    recall = {}                                       # direction -> {dim:[], cont:[], sign:[], asym:[]}
    bd = {}                                            # direction -> grid
    for lang, txt in texts.items():
        # PCA fit on the union of the two modalities (shared cross-modal space)
        mu, V, S = pca_fit(torch.cat([img, txt], 0))
        N = img.shape[0]; gold = torch.arange(N, device=dev)
        for direction, (q_src, g_src) in [("T2I", (txt, img)), ("I2T", (img, txt))]:
            key = f"{direction}/{lang}"
            recall[key] = {"dim": [], "cont": [], "sign": [], "sign_mc": [], "asym": []}
            for d in dims:
                qp = project(q_src, mu, V, d); gp = project(g_src, mu, V, d)
                row = {"backbone": name, "lang": lang, "direction": direction, "dim": d}
                for mode, col in [("continuous", "r10_cont"), ("sign", "r10_sign"),
                                  ("sign_mc", "r10_sign_mc"), ("asym", "r10_asym")]:
                    qr, gr = codes_for(qp, gp, mode)
                    row[col] = r_at_k(qr @ gr.t(), gold)
                dim_rows.append(row)
                recall[key]["dim"].append(d)
                recall[key]["cont"].append(row["r10_cont"])
                recall[key]["sign"].append(row["r10_sign"])
                recall[key]["sign_mc"].append(row["r10_sign_mc"])
                recall[key]["asym"].append(row["r10_asym"])
            # bits×dims grid (continuous / 2-bit / 1-bit) on this direction+lang
            bdkey = key
            bd[bdkey] = {"dims": dims, "bpd": ["float", "twobit", "sign"], "grid": [], "total_bits": []}
            for mode, bpd in [("continuous", 32), ("twobit", 2), ("sign", 1)]:
                rrow, trow = [], []
                for d in dims:
                    qp = project(q_src, mu, V, d); gp = project(g_src, mu, V, d)
                    qr, gr = codes_for(qp, gp, mode)
                    r = r_at_k(qr @ gr.t(), gold)
                    rrow.append(r); trow.append(d * bpd)
                    bd_rows.append({"backbone": name, "lang": lang, "direction": direction,
                                    "dim": d, "bits_per_dim": bpd, "total_bits": d * bpd, "r10": r})
                bd[bdkey]["grid"].append(rrow); bd[bdkey]["total_bits"].append(trow)
            print(f"  [{name}] {key}: cont {recall[key]['cont'][-1]} / sign {recall[key]['sign'][-1]} "
                  f"@full{dims[-1]}  | low-d sign@8={recall[key]['sign'][2] if len(recall[key]['sign'])>2 else 'na'}",
                  flush=True)
    return dim_rows, bd_rows, recall, bd


def spectrum_and_pc(name, bb, lang="en", max_pc=256):
    """Singular-value spectrum, eff-dim, per-PC retrieval contribution (variance vs retrieval order)."""
    D = bb["D"]; img = bb["img"].to(dev); txt = bb[lang].to(dev)
    mu, V, S = pca_fit(torch.cat([img, txt], 0))
    eig = (S ** 2)                                     # ∝ variance per component
    eig = eig / eig.sum()
    cumvar = torch.cumsum(eig, 0)
    pr = (S ** 2).sum() ** 2 / (S ** 4).sum()          # participation ratio
    def first_ge(t):
        idx = (cumvar >= t).nonzero()
        return int(idx[0].item()) + 1 if len(idx) else D
    out = {"backbone": name, "lang": lang, "D": D,
           "sv": S.cpu().tolist(), "var_frac": eig.cpu().tolist(), "cumvar": cumvar.cpu().tolist(),
           "participation_ratio": round(float(pr.item()), 2),
           "d90": first_ge(0.90), "d95": first_ge(0.95), "d99": first_ge(0.99),
           "pc": {}}
    N = img.shape[0]; gold = torch.arange(N, device=dev)
    P = min(max_pc, D)
    qp_all = project(txt, mu, V, P); gp_all = project(img, mu, V, P)   # T2I
    # single-PC R@10 (rank-1 score q_i*g_i)
    indiv = []
    for i in range(P):
        sc = qp_all[:, i].unsqueeze(1) * gp_all[:, i].unsqueeze(0)   # rank-1 score (N,N)
        indiv.append(r_at_k(sc, gold))
    indiv_t = torch.tensor(indiv)
    # cumulative R@10, unnormalized dot, snapshotted on a grid
    KS = sorted(set([k for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256] if k <= P] + [P]))
    var_order = list(range(P))
    retr_order = list(np.argsort(-indiv_t.numpy()))
    def cum_curve(order):
        # incremental cumulative dot over PCs in `order`
        S_mat = torch.zeros(N, N, device=dev); curve = []; ks = set(KS); col = 0
        for step, j in enumerate(order, 1):
            S_mat += qp_all[:, j].unsqueeze(1) * gp_all[:, j].unsqueeze(0)
            if step in ks:
                curve.append(r_at_k(S_mat, gold))
        return curve
    var_cum = cum_curve(var_order); retr_cum = cum_curve(retr_order)
    from scipy.stats import spearmanr
    # retrieval-rank vs variance-rank correlation
    retr_rank = np.empty(P, int); retr_rank[np.argsort(-indiv_t.numpy())] = np.arange(P)
    sp = spearmanr(np.arange(P), retr_rank).correlation
    out["pc"] = {"P": P, "indiv_r10": indiv, "KS": KS,
                 "var_order_cumr10": var_cum, "retr_order_cumr10": retr_cum,
                 "retr_order_idx": [int(x) for x in retr_order[:64]],
                 "spearman_var_vs_retr": round(float(sp), 3),
                 "ceiling_r10": r_at_k((l2(qp_all) @ l2(gp_all).t()), gold)}
    print(f"  [{name}] spectrum: PR {out['participation_ratio']} d95 {out['d95']} "
          f"| PC var-vs-retr Spearman {out['pc']['spearman_var_vs_retr']} ceiling {out['pc']['ceiling_r10']}",
          flush=True)
    return out


HEADS = {  # backbone -> (head_ckpt, test_pt, img_key, txt_key)
    "so400m": ("/tmp/ft_ko_113.pt", "/tmp/emb_cache.pt", "img", "txt"),
    "siglip2-base": ("/tmp/bb_siglip2-base_head.pt", "/tmp/bb_siglip2-base_test.pt", "img", "en"),
    "altclip-m18": ("/tmp/bb_altclip-m18_head.pt", "/tmp/bb_altclip-m18_test.pt", "img", "en"),
}


def head_concentration(name):
    """Trained matryoshka HEAD continuous R@10 at each nested dim (T2I, EN). The decisive
    raw-vs-head comparison: does the head pack retrieval info into far fewer dims than raw PCA?"""
    sys.path.insert(0, REPO)
    from src.models.nested_hash_layer import NestedHashLayer
    ck_path, test_path, ik, tk = HEADS[name]
    ck = torch.load(ck_path, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]; embed = ck["embed"]; hidden = ck["hidden"]
    def mk(sd):
        m = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval(); m.load_state_dict(sd); return m
    img_h = mk(ck["img_h"]); txt_h = mk(ck["txt_h"])
    test = torch.load(test_path, map_location="cpu")
    if "test" in test: test = test["test"]
    img = l2(test[ik].float().to(dev)); txt = l2(test[tk].float().to(dev))
    N = img.shape[0]; gold = torch.arange(N, device=dev)
    def z_at(head, x, b):
        raw = head.hash_head(x)[:, :b]
        return torch.nn.functional.normalize(head.batch_norms[bits.index(b)](raw), p=2, dim=1)
    cont, sign = [], []
    with torch.no_grad():
        for b in bits:
            zg = z_at(img_h, img, b); zq = z_at(txt_h, txt, b)
            cont.append(r_at_k(zq @ zg.t(), gold))
            sign.append(r_at_k(torch.sign(zq) @ torch.sign(zg).t(), gold))
    print(f"  [HEAD {name}] bits {bits}: cont {cont} | sign {sign}", flush=True)
    return {"dim": bits, "cont": cont, "sign": sign}


def xm_lang_saturation():
    """XM3600 so400m: per-language PCA dim-sweep -> saturation point (text->image)."""
    try:
        xm = torch.load("/tmp/xm_so400m_lc.pt", map_location="cpu")
    except Exception:
        xm = torch.load("/tmp/xm_so400m.pt", map_location="cpu")
    img = xm["img_emb"].float().to(dev)
    LATIN = set("cs da de en es fi fil fr hr hu id it mi nl no pl pt quz ro sv sw tr vi".split())
    NONLATIN = set("ar bn el fa he hi ja ko ru te th uk zh".split())
    dims = [d for d in DIMS_FULL if d <= img.shape[1]] + [img.shape[1]]
    dims = sorted(set(dims))
    rows = []; per_lang = {}
    for lang, pack in xm["per_lang"].items():
        txt = pack["text_emb"].float().to(dev); gold = torch.as_tensor(pack["gold"]).long().to(dev)
        mu, V, S = pca_fit(torch.cat([img, txt], 0))
        r10 = []
        for d in dims:
            qp = project(txt, mu, V, d); gp = project(img, mu, V, d)
            r = r_at_k(l2(qp) @ l2(gp).t(), gold)
            r10.append(r); rows.append({"lang": lang, "dim": d, "r10": r})
        full = r10[-1]
        # saturation dim = smallest d reaching 95% of full
        sat = dims[-1]
        for d, r in zip(dims, r10):
            if full > 0 and r >= 0.95 * full:
                sat = d; break
        script = "latin" if lang in LATIN else ("non-latin" if lang in NONLATIN else "other")
        per_lang[lang] = {"r10": r10, "full_r10": full, "sat_dim": sat, "script": script}
    print(f"  XM3600: {len(per_lang)} langs, dims {dims}", flush=True)
    return dims, rows, per_lang


# ───────────────────────── run ─────────────────────────
def main():
    bbs = load_backbones()
    print("backbones:", {k: v["D"] for k, v in bbs.items()}, flush=True)

    all_dim_rows, all_bd_rows = [], []
    recall_payload, bd_payload, spec_payload = {}, {}, {}
    for name, bb in bbs.items():
        dr, bdr, rec, bd = sweep_backbone(name, bb)
        all_dim_rows += dr; all_bd_rows += bdr
        recall_payload[name] = rec; bd_payload[name] = bd
        spec_payload[name] = spectrum_and_pc(name, bb, lang="en")
        try:
            recall_payload[name]["HEAD/en"] = head_concentration(name)
        except Exception as e:
            print(f"  HEAD {name} skip:", e)

    # XM3600 multilingual saturation
    try:
        xm_dims, xm_rows, xm_per_lang = xm_lang_saturation()
    except Exception as e:
        print("XM3600 skip:", e); xm_dims, xm_rows, xm_per_lang = [], [], {}

    # ── CSVs ──
    def write_csv(path, rows):
        if not rows: return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_csv(os.path.join(OUTp, "gapB_dim_sweep.csv"), all_dim_rows)
    write_csv(os.path.join(OUTp, "gapB_bits_dims.csv"), all_bd_rows)
    effrows = [{"backbone": s["backbone"], "lang": s["lang"], "D": s["D"],
                "participation_ratio": s["participation_ratio"], "d90": s["d90"],
                "d95": s["d95"], "d99": s["d99"],
                "ceiling_r10": s["pc"]["ceiling_r10"],
                "spearman_var_vs_retr": s["pc"]["spearman_var_vs_retr"]} for s in spec_payload.values()]
    write_csv(os.path.join(OUTp, "gapB_effdim.csv"), effrows)
    pcrows = []
    for s in spec_payload.values():
        for i, (v, c) in enumerate(zip(s["var_frac"][:s["pc"]["P"]], s["cumvar"][:s["pc"]["P"]])):
            pcrows.append({"backbone": s["backbone"], "pc_index": i, "var_frac": round(v, 6),
                           "cumvar": round(c, 6), "indiv_r10": s["pc"]["indiv_r10"][i]})
    write_csv(os.path.join(OUTp, "gapB_pc_contribution.csv"), pcrows)
    write_csv(os.path.join(OUTp, "gapB_lang_saturation.csv"), xm_rows)

    # ── JSON for HTML ──
    json.dump({"backbones": list(recall_payload.keys()), "dims_full": DIMS_FULL,
               "data": recall_payload,
               "note": "PCA dim-sweep of RAW frozen embeddings. cont=cosine, sign=Hamming(b·b), asym=z_q·b_g. R@10."},
              open(os.path.join(OUTv, "gapB_dim_recall.json"), "w"))
    json.dump({"backbones": list(spec_payload.keys()), "data": spec_payload,
               "note": "SV spectrum + cumvar + participation ratio; per-PC single R@10; variance-order vs retrieval-order cumulative R@10 (unnormalized dot). T2I, EN."},
              open(os.path.join(OUTv, "gapB_spectrum.json"), "w"))
    json.dump({"backbones": list(bd_payload.keys()), "data": bd_payload,
               "note": "bits×dims grid: rows=[float,2-bit,1-bit]; cols=dims; R@10 + total_bits for equal-budget contours."},
              open(os.path.join(OUTv, "gapB_bits_dims.json"), "w"))
    json.dump({"dims": xm_dims, "langs": list(xm_per_lang.keys()), "per_lang": xm_per_lang,
               "note": "XM3600 so400m (lowercased) text->image PCA dim-sweep; sat_dim = smallest d at >=95% of full-dim R@10."},
              open(os.path.join(OUTv, "gapB_lang_sat.json"), "w"))
    print("GAPB_DONE -> paper/gapB_*.csv, viz/data/gapB_*.json", flush=True)


if __name__ == "__main__":
    main()
