"""TEXT GATE — loss-composition study for cross-modal 1-bit hashing (AAAI).

Trains a text head txt_h' mapping NATIVE e5-small text embeddings into the FROZEN
SigLIP2-So400m 1024-bit code space (anchor = ft113 img_h, the exact head behind
index.bin). Compares THREE loss COMPOSITIONS holding the same loss INGREDIENTS:

    align   = InfoNCE(tanh(z_k/T), tanh(a_k))   [match the anchor code]
    quant   = EAQL(tanh(z_k/T))                 [push away from 0]
    margin  = mean(relu(m - z_k * b_k))         [pre-sign agrees w/ anchor bit by m]
    nesting = LCS(tanh(z_k/T))                  [long->short self-distill]

  (W) weighted-sum : L = align + wq*quant + wm*margin + wl*nesting, T=1 fixed.
  (C) curriculum   : T annealed T0->1 (cosine); wq,wm ramp 0->target over first 60%.
  (U) unified      : U = mean(relu(m - z_k*b_k)^2); L = align + wu*U + wl*nesting, T=1.

The head reuses the SAME NestedHashLayer modules (hash_head + batch_norms) so the
trained params == txt_h' exactly. z_k = L2norm(BN_k(slice(hash_head(e5)))) is the
PRE-SIGN normalized vector (NestedHashLayer does not expose it, so we replicate its
forward); a_k = same from the frozen img_h; b_k = sign(a_k).detach() = target bits.

Eval: gallery = frozen img_h(test_img) codes; query = txt_h'(e5_test) codes; faiss
IndexBinaryFlat at nested prefix bits {64,128,256,512,1024}. R@{1,5,10} EN & KO.
Diagnostics (1024-bit): sign-margin |z|, noise flip%, paired cosine(z,a).

Run on DGX (single sweep):
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/loss_composition.py --sweep \
      --out /tmp/loss_comp.csv 2>&1 | tee /tmp/loss_comp.log
Single run (smoke):
  .venv/bin/python web/loss_composition.py --comp W --epochs 1 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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

HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
E5_CACHE = os.environ.get("E5_CACHE", "/tmp/e5_train_emb.pt")
STUDENT = os.environ.get("STUDENT", "intfloat/multilingual-e5-small")
E5_TEST_EN = os.environ.get("E5_TEST_EN", "/tmp/e5_test_en.pt")
E5_TEST_KO = os.environ.get("E5_TEST_KO", "/tmp/e5_test_ko.pt")
MAXLEN = int(os.environ.get("MAXLEN", "64"))
EVAL_BITS = [64, 128, 256, 512, 1024]
dev = "cuda" if torch.cuda.is_available() else "cpu"
_ID = re.compile(r"_0*(\d+)\.jpg")
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _cocoid(p):
    m = _ID.search(p)
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------- #
#  Replicate NestedHashLayer.forward but EXPOSE pre-sign z_k.
#  Mirrors src/models/nested_hash_layer.py exactly:
#     raw = hash_head(x); sliced = raw[:, :length]
#     z_k = F.normalize(bn_k(sliced), p=2, dim=1)   # the "normalized" pre-sign
#     continuous = tanh(z_k);  binary = sign(z_k)
# --------------------------------------------------------------------------- #
def head_z(head: NestedHashLayer, x: torch.Tensor, only_bits=None):
    """Return list of pre-sign normalized z_k (one per bit), reusing head modules.
    If only_bits is given, return z_k only for those bit lengths (saves compute).
    Still runs ALL BatchNorms (they need running stats), but only returns requested."""
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
#  Data
# --------------------------------------------------------------------------- #
def build_train_data():
    """(e5_text (N,384) L2, so400m_img (N,1152) L2) aligned EN+KO pairs.

    Imports web.headadapt_train.build_data (the reference pairing — hits e5 cache).
    """
    from web.headadapt_train import build_data
    return build_data()


class E5:
    def __init__(self, name, device):
        from transformers import AutoModel, AutoTokenizer
        self.m = AutoModel.from_pretrained(name).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.device = device

    def embed(self, strings, maxlen=MAXLEN, batch=256):
        out = []
        with torch.no_grad():
            for s in range(0, len(strings), batch):
                t = self.tok(strings[s:s + batch], padding="max_length", max_length=maxlen,
                             truncation=True, return_tensors="pt")
                o = self.m(t["input_ids"].to(self.device),
                           t["attention_mask"].to(self.device)).last_hidden_state
                msk = t["attention_mask"].to(self.device).unsqueeze(-1).float()
                e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
                out.append(F.normalize(e, dim=1).float().cpu())
        return torch.cat(out, 0)


def build_test_e5():
    """Compute (once, cached) native e5 test embeddings for EN & KO test captions,
    aligned to emb_cache test ids. Returns (e5_en (5000,384), e5_ko (5000,384),
    img_emb (5000,1152), te_ids list)."""
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    img_emb = EC["test"]["img"].float()
    en_caps = [str(c) for c in EC["test"]["captions"]]
    ko_full = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line)
        ko_full[_cocoid(e["image_path"])] = e.get("captions", [])
    ko_caps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]

    def _load_or_compute(path, caps):
        if os.path.exists(path):
            t = torch.load(path, map_location="cpu")
            if t.shape[0] == len(caps):
                return t
        enc = build_test_e5._enc
        if enc is None:
            enc = E5(STUDENT, dev)
            build_test_e5._enc = enc
        t = enc.embed(caps)
        torch.save(t, path)
        return t

    build_test_e5._enc = None
    e5_en = _load_or_compute(E5_TEST_EN, en_caps)
    e5_ko = _load_or_compute(E5_TEST_KO, ko_caps)
    return e5_en, e5_ko, img_emb, te_ids


# --------------------------------------------------------------------------- #
#  Loss terms
# --------------------------------------------------------------------------- #
def margin_value(z_list, rel=0.1):
    """Relative margin m = rel * mean_over_bits(std(z_k)) for the batch."""
    with torch.no_grad():
        return rel * float(torch.stack([z.std() for z in z_list]).mean())


def compute_loss(comp, z_list, a_list, b_list, contrastive, eaql, lcs,
                 T, wq, wm, wl, wu, m_abs):
    """Return scalar loss for the chosen composition.
    z_list: trainable head pre-sign (per bit). a_list: frozen anchor pre-sign.
    b_list: target bits sign(a).detach(). T: tanh temperature. m_abs: absolute margin."""
    align = torch.zeros((), device=z_list[0].device)
    quant = torch.zeros((), device=z_list[0].device)
    margin = torch.zeros((), device=z_list[0].device)
    nb = len(z_list)
    tz = [torch.tanh(z / T) for z in z_list]
    ta = [torch.tanh(a) for a in a_list]
    for k in range(nb):
        align = align + contrastive(tz[k], ta[k])
        quant = quant + eaql(tz[k])
        if comp == "U":
            margin = margin + (F.relu(m_abs - z_list[k] * b_list[k]) ** 2).mean()
        else:
            margin = margin + F.relu(m_abs - z_list[k] * b_list[k]).mean()
    align = align / nb
    quant = quant / nb
    margin = margin / nb
    nesting = lcs(tz)
    if comp == "U":
        return align + wu * margin + wl * nesting
    return align + wq * quant + wm * margin + wl * nesting


# --------------------------------------------------------------------------- #
#  Train one configuration
# --------------------------------------------------------------------------- #
LOSS_BITS = set(EVAL_BITS)  # only compute loss on eval bits (skip 8/16/32 for speed)


def train_one(comp, setting, hp, e5_tr, img_tr, bits, hidden, img_h,
              epochs=15, bs=512, seed=42,
              wq=None, wm_rel=0.1, wm=None, wl=None, wu=None, m_rel=0.1,
              T0=4.0, ramp_frac=0.6):
    """Train txt_h' under a composition; return the trained head."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    e5_dim = e5_tr.shape[1]
    N = e5_tr.shape[0]
    txt_h = NestedHashLayer(e5_dim, hidden, bits, hp["dropout"]).to(dev).train()

    contrastive = CrossModalContrastiveLoss(hp["temperature"]).to(dev)
    eaql = EAQLLoss().to(dev)
    lcs = LCSSelfDistillationLoss().to(dev)

    opt = torch.optim.AdamW(txt_h.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
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
            with torch.no_grad():
                a_list = head_z(img_h, img_tr[idx].to(dev), only_bits=LOSS_BITS)
                b_list = [torch.sign(a).detach() for a in a_list]
            z_list = head_z(txt_h, e5_tr[idx].to(dev), only_bits=LOSS_BITS)
            prog = g / max(steps, 1)
            if comp == "C":
                T = 1.0 + (T0 - 1.0) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
                ramp = min(1.0, prog / ramp_frac)
                cur_wq, cur_wm, cur_wu = wq * ramp, wm * ramp, wu * ramp
            else:
                T = 1.0
                cur_wq, cur_wm, cur_wu = wq, wm, wu
            m_abs = margin_value(z_list, rel=m_rel)
            loss = compute_loss(comp, z_list, a_list, b_list, contrastive, eaql, lcs,
                                T, cur_wq, cur_wm, wl, cur_wu, m_abs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            g += 1
    txt_h.eval()
    return txt_h


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



def eval_run(txt_h, img_h, bits, e5_en, e5_ko, img_emb, te_ids):
    """Return per-bit dict of recalls EN/KO. Gallery codes from img_h; queries txt_h'."""
    # only need 1024-bit code; eval bits are prefix slices of it
    with torch.no_grad():
        gal_full = (head_z(img_h, img_emb.to(dev), only_bits={1024})[0] > 0).cpu().numpy().astype(np.uint8)
        en_full = (head_z(txt_h, e5_en.to(dev), only_bits={1024})[0] > 0).cpu().numpy().astype(np.uint8)
        ko_full = (head_z(txt_h, e5_ko.to(dev), only_bits={1024})[0] > 0).cpu().numpy().astype(np.uint8)
    res = {}
    for b in EVAL_BITS:
        gal = pack_bits(gal_full[:, :b])
        ix = _faiss(gal, b)
        row = {}
        for lang, q_full in (("EN", en_full), ("KO", ko_full)):
            q = pack_bits(q_full[:, :b])
            for k in (1, 5, 10):
                row[f"R{k}_{lang}"] = _recall(ix, q, te_ids, k)
        res[b] = row
    return res, en_full


def diagnostics(txt_h, img_h, bits, e5_en, img_emb):
    """1024-bit diagnostics on EN test: sign-margin |z|, noise flip%, paired cosine."""
    with torch.no_grad():
        z = head_z(txt_h, e5_en.to(dev), only_bits={1024})[0]    # (N,1024) pre-sign text
        a = head_z(img_h, img_emb.to(dev), only_bits={1024})[0]  # paired anchor pre-sign
    az = z.abs()
    margin_mean = float(az.mean())
    margin_median = float(az.median())
    margin_p10 = float(torch.quantile(az.flatten().float(), 0.10))
    # paired cosine(z, a) — text vs its own image anchor (continuous alignment)
    cos_pair = float(F.cosine_similarity(z, a, dim=1).mean())
    # flip% under input noise
    e5 = e5_en.to(dev)
    norm = e5.norm(dim=1, keepdim=True)
    sigma = 0.05 * norm / math.sqrt(e5.shape[1])
    clean_sign = torch.sign(z)
    flips = []
    g = torch.Generator(device=dev).manual_seed(123)
    with torch.no_grad():
        for _ in range(5):
            noise = torch.randn(e5.shape, generator=g, device=dev) * sigma
            zn = head_z(txt_h, F.normalize(e5 + noise, dim=1), only_bits={1024})[0]
            flips.append(float((torch.sign(zn) != clean_sign).float().mean()))
    flip_pct = round(100 * float(np.mean(flips)), 3)
    return {
        "margin_mean": round(margin_mean, 4),
        "margin_median": round(margin_median, 4),
        "margin_p10": round(margin_p10, 4),
        "flip_pct": flip_pct,
        "cos_pair": round(cos_pair, 4),
    }


# --------------------------------------------------------------------------- #
#  Grids
# --------------------------------------------------------------------------- #
def build_grid(hp):
    """Return list of run configs: (comp, setting_id, kwargs, seeds).
    Trimmed grid (~21 runs total incl variance) for feasible wall-time."""
    runs = []
    # NOTE: gate-critical compositions (C, U) are emitted FIRST so a partial/interrupted
    # sweep still yields the C-vs-U-vs-W signal; the W sensitivity grid follows.
    # (C) curriculum: default + T0 in {2,4}
    for T0 in (2.0, 4.0):
        runs.append(("C", f"C_T0{T0:g}", {"T0": T0}, [42]))
    # (U) unified: grid over (wu, m_rel) — 6 points
    for wu in (0.1, 0.2, 0.4):
        for m_rel in (0.05, 0.1):
            runs.append(("U", f"U_wu{wu:g}_m{m_rel:g}", {"wu": wu, "m_rel": m_rel}, [42]))
    # (W) weight-sensitivity grid over (wm, wq): 3x3=9 points (generous for fairness)
    wm_pts = [0.0, 0.1, 0.4]
    wq_pts = [0.06, hp["quant"], 0.25]
    for wq in wq_pts:
        for wm in wm_pts:
            sid = f"W_wq{wq:g}_wm{wm:g}"
            runs.append(("W", sid, {"wq": wq, "wm": wm}, [42]))
    return runs


CSV_COLS = ["comp", "setting", "seed", "bit", "R1_EN", "R5_EN", "R10_EN",
            "R1_KO", "R5_KO", "R10_KO", "margin_mean", "margin_p10",
            "flip_pct", "cos_pair"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comp", choices=["W", "C", "U"], default="W")
    p.add_argument("--setting", default="single")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--out", default="/tmp/loss_comp.csv")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny: 1 epoch, no eval persistence")
    # single-run knobs
    p.add_argument("--wq", type=float, default=None)
    p.add_argument("--wm", type=float, default=None)
    p.add_argument("--wl", type=float, default=None)
    p.add_argument("--wu", type=float, default=None)
    p.add_argument("--m_rel", type=float, default=0.1)
    p.add_argument("--T0", type=float, default=4.0)
    args = p.parse_args()

    t0 = time.perf_counter()
    hp = json.load(open("/tmp/hp_results.json"))["best_params"]
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    hidden, embed = ck["hidden"], ck["embed"]

    img_h = NestedHashLayer(embed, hidden, bits, 0.0).to(dev).eval()
    img_h.load_state_dict(ck["img_h"])
    for pa in img_h.parameters():
        pa.requires_grad_(False)

    e5_tr, img_tr = build_train_data()
    if args.smoke:
        e5_tr, img_tr = e5_tr[:8192], img_tr[:8192]
    e5_en, e5_ko, img_emb, te_ids = build_test_e5()
    print(f"[setup] train N={e5_tr.shape[0]:,} test EN/KO=5000 bits={bits} "
          f"({(time.perf_counter()-t0)/60:.1f}min)", flush=True)

    rows = []

    def run(comp, sid, kw, seed, epochs):
        rt = time.perf_counter()
        txt_h = train_one(comp, sid, hp, e5_tr, img_tr, bits, hidden, img_h,
                          epochs=epochs, bs=args.bs, seed=seed,
                          wq=kw.get("wq"), wm=kw.get("wm"), wl=kw.get("wl"),
                          wu=kw.get("wu"), m_rel=kw.get("m_rel", 0.1),
                          T0=kw.get("T0", 4.0))
        res, _ = eval_run(txt_h, img_h, bits, e5_en, e5_ko, img_emb, te_ids)
        diag = diagnostics(txt_h, img_h, bits, e5_en, img_emb)
        # free GPU memory from this run's head before next run
        del txt_h
        torch.cuda.empty_cache()
        for b in EVAL_BITS:
            r = res[b]
            rows.append({
                "comp": comp, "setting": sid, "seed": seed, "bit": b,
                **{k: r[k] for k in ("R1_EN", "R5_EN", "R10_EN", "R1_KO", "R5_KO", "R10_KO")},
                "margin_mean": diag["margin_mean"], "margin_p10": diag["margin_p10"],
                "flip_pct": diag["flip_pct"], "cos_pair": diag["cos_pair"],
            })
        # incremental durable write: rewrite full CSV after each config (partial results survive a drop)
        import csv as _csv
        with open(args.out, "w", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=CSV_COLS); _w.writeheader(); _w.writerows(rows)
        r1024 = res[1024]
        print(f"[run] {comp} {sid} s{seed}: 1024 R@10 EN {r1024['R10_EN']} KO {r1024['R10_KO']} "
              f"| flip {diag['flip_pct']}% margin {diag['margin_mean']} cos {diag['cos_pair']} "
              f"({(time.perf_counter()-rt)/60:.1f}min)", flush=True)
        return r1024["R10_EN"], r1024["R10_KO"]

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
                en, ko = run(comp, sid, kw, seed, args.epochs)
                score = en + ko
                if score > best[comp][0]:
                    best[comp] = (score, (sid, kw))
        # variance: seeds 0 & 1 for the best setting of each comp
        for comp in ("W", "C", "U"):
            sid, kw = best[comp][1]
            for seed in (0, 1):
                run(comp, sid, kw, seed, args.epochs)
        print(f"[sweep] best: " + json.dumps({c: best[c][1][0] for c in best}), flush=True)

    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {len(rows)} rows -> {args.out} ({(time.perf_counter()-t0)/60:.1f}min)",
          flush=True)


if __name__ == "__main__":
    main()
