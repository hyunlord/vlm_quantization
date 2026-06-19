"""Ext② diagnostic — maxlen sensitivity + German/so400m-text anomaly. EVAL ONLY, 1024-bit,
no retraining / no index or head change. Reuses the frozen ft113 gallery + Ext① offline heads.

Resolves two open questions from web/PAPER_MULTILING.md:

(A) Offline OOD drop = truncation artifact? The offline heads were TRAINED at maxlen=64 on (short)
    COCO captions — which never truncate at 64 — so the head learned a *full-caption* mean-pool. XM3600
    captions can exceed 64 tokens, so eval@64 feeds the head a TRUNCATED mean-pool (out of its training
    distribution). e5/MiniLM/e5-base support 512 positions, so we re-encode at maxlen {64,128,256} and
    check whether long-caption languages recover (short-caption langs are identical across maxlen under
    mean-pool, so this isolates truncation).

(B) German/te/th/hi so400m-text anomaly. so400m text is structurally capped at 64 positions
    (text max_position_embeddings=64) — it CANNOT see >64 tokens, so a server longer-maxlen recovery is
    impossible. Instead we (1) tabulate per-language token lengths under the Gemma (SigLIP) vs XLM-R
    (e5/MiniLM) tokenizers, and (2) stratify float/server R@10 by caption length (<=64 vs >64 Gemma
    tokens). If the >64 bucket collapses, the anomaly is the 64-token cap x tokenizer (in)efficiency,
    not a semantic so400m weakness or a bug.

Outputs: paper/multiling_toklen.csv, paper/multiling_lenstrat.csv, paper/multiling_maxlen.csv.

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_multiling_maxlen.py --data data/xm3600
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, MODEL as SIGLIP_MODEL, Encoder, pack_bits  # noqa: E402
from web.paper_multiling_run import (  # noqa: E402  (reuse exact eval contract)
    OFFLINE, REPR_LANGS, faiss_bin, load_xm3600, offline_emb, recall_ks,
    so400m_image_emb, so400m_text_emb,
)

PAPER = Path(REPO) / "paper"
MAXLENS = (64, 128, 256)


def tok_lengths(tok, strings, batch=512, cap=512):
    """True token length per string. XLM-R returns attention_mask; the SigLIP/Gemma tokenizer does
    NOT (fixed-length padding, no mask) — fall back to counting non-pad input_ids."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else getattr(tok, "eos_token_id", None)
    out = []
    for s in range(0, len(strings), batch):
        t = tok(strings[s:s + batch], padding="max_length", max_length=cap, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        if am is not None:
            out.extend(int(x) for x in am.sum(1).tolist())
        elif pad is not None:
            out.extend(int(x) for x in (t["input_ids"] != pad).sum(1).tolist())
        else:
            out.extend([int(t["input_ids"].shape[1])] * t["input_ids"].shape[0])
    return out


def hits_at10_float(txt_norm, img_norm, cap_img):
    topk = (txt_norm @ img_norm.t()).topk(10, dim=1).indices.numpy()
    return np.array([cap_img[i] in topk[i] for i in range(len(cap_img))], dtype=bool)


def hits_at10_codes(gal_ix, codes, cap_img):
    _, I = gal_ix.search(codes, 10)
    return np.array([cap_img[i] in I[i] for i in range(len(cap_img))], dtype=bool)


def bucket_r10(hits, lens, thr=64):
    lens = np.asarray(lens)
    out = {}
    for name, mask in (("all", np.ones(len(hits), bool)), ("le64", lens <= thr), ("gt64", lens > thr)):
        n = int(mask.sum())
        out[name] = (round(100 * float(hits[mask].mean()), 2) if n else None, n)
    return out


@torch.no_grad()
def offline_emb_dyn(model, tok, strings, prefix, dev, maxlen, batch=256):
    """Mean-pool + L2 with DYNAMIC padding (pad to batch-longest). Numerically identical to
    paper_multiling_run.offline_emb (attention-masked mean-pool ignores pad positions, and real-token
    outputs don't depend on pad count), but far faster at large maxlen since short captions are not
    padded to maxlen. `maxlen` only sets the truncation point."""
    import torch.nn.functional as F
    out = []
    for s in range(0, len(strings), batch):
        txt = [prefix + x for x in strings[s:s + batch]]
        t = tok(txt, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        o = model(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
        msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
        e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        out.append(F.normalize(e, dim=1).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xm3600")
    ap.add_argument("--langs", default="", help="comma list; default = all present")
    ap.add_argument("--limit-images", type=int, default=0, help="smoke: cap gallery size")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dev == "cpu" and "fp32" or args.dtype]
    langs = [x for x in args.langs.split(",") if x] or None

    image_paths, per_lang = load_xm3600(args.data, langs, args.limit_images)
    all_langs = sorted(per_lang.keys())
    n_img = len(image_paths)
    print(f"[mx] images={n_img} langs={len(all_langs)}", flush=True)

    enc = Encoder()
    PAPER.mkdir(parents=True, exist_ok=True)

    # ---- so400m backbone: gallery + server@64 + float + Gemma token lengths + length-strat ----
    from transformers import AutoModel, AutoTokenizer
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    except Exception:
        from transformers import GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
        processor = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(SIGLIP_MODEL),
                                    tokenizer=GemmaTokenizer.from_pretrained(SIGLIP_MODEL))
    gemma_tok = processor.tokenizer
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dtype).to(dev).eval()

    emb_cache = f"/tmp/xm3600_imgemb_{n_img}.npy"  # XM3600 image decode is slow (~8/s); cache the so400m image emb
    if os.path.exists(emb_cache):
        img_emb = torch.from_numpy(np.load(emb_cache))
        print(f"[mx] image emb from cache {emb_cache} {tuple(img_emb.shape)}", flush=True)
    else:
        img_emb = so400m_image_emb(bb, processor, image_paths, dev, dtype)
        np.save(emb_cache, img_emb.numpy())
    gal = enc.image_codes_packed(img_emb.numpy())
    gal_ix = faiss_bin(gal, CODE_BITS)
    img_norm = F.normalize(img_emb, dim=1)
    print(f"[mx] gallery {tuple(img_emb.shape)} ready", flush=True)

    gemma_lens = {}            # lang -> per-caption Gemma token length
    lenstrat = {}              # (which, lang) -> bucket dict   which in {float, server}
    for L in all_langs:
        caps, cap_img = per_lang[L]
        gl = tok_lengths(gemma_tok, caps)
        gemma_lens[L] = gl
        te = so400m_text_emb(bb, gemma_tok, caps, dev, dtype, maxlen=64)  # server: capped at 64
        hits_srv = hits_at10_codes(gal_ix, pack_bits(enc._codes_pm1(enc.txt_h, te)), cap_img)
        hits_flt = hits_at10_float(F.normalize(te, dim=1), img_norm, cap_img)
        lenstrat[("server", L)] = bucket_r10(hits_srv, gl)
        lenstrat[("float", L)] = bucket_r10(hits_flt, gl)
        b = lenstrat[("server", L)]
        print(f"[mx] {L}: server all {b['all'][0]} | <=64 {b['le64'][0]}(n{b['le64'][1]}) | >64 {b['gt64'][0]}(n{b['gt64'][1]})", flush=True)
    del bb
    gc.collect()
    torch.cuda.empty_cache() if dev == "cuda" else None

    # ---- XLM-R token lengths (e5/MiniLM share the XLM-R sentencepiece tokenizer) ----
    xlmr_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    xlmr_lens = {L: tok_lengths(xlmr_tok, per_lang[L][0]) for L in all_langs}

    # ---- offline maxlen sweep {64,128,256} ----
    sweep = {}  # (label, lang, maxlen) -> R@10
    for spec in OFFLINE:
        head = spec["head"]
        if not os.path.exists(head):
            print(f"[mx] SKIP {spec['label']} (no head)", flush=True)
            continue
        ck = torch.load(head, map_location="cpu")
        student, prefix = ck["student"], (ck.get("prefix") or "")
        bits = [int(b) for b in ck["bits"]]
        obi = bits.index(CODE_BITS)
        txt_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0)
        txt_h.load_state_dict(ck["txt_h"])
        txt_h.to(dev).eval()
        m = AutoModel.from_pretrained(student).to(dev).eval()
        tok = AutoTokenizer.from_pretrained(student)
        print(f"[mx] === {spec['label']} ({student}) ===", flush=True)
        for ml in MAXLENS:
            for L in all_langs:
                caps, cap_img = per_lang[L]
                e = offline_emb_dyn(m, tok, caps, prefix, dev, ml)
                with torch.no_grad():
                    codes = pack_bits(txt_h(e.to(dev))[obi]["binary"].cpu().numpy())
                sweep[(spec["label"], L, ml)] = recall_ks(gal_ix, codes, cap_img)[10]
            probe = [L for L in ("ko", "de", "th", "en") if L in all_langs] or all_langs[:3]
            print(f"[mx]   maxlen={ml}: " + " ".join(f"{L} {sweep[(spec['label'], L, ml)]}" for L in probe), flush=True)
        del m, txt_h
        gc.collect()
        torch.cuda.empty_cache() if dev == "cuda" else None

    # ---- write CSVs ----
    with open(PAPER / "multiling_toklen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "n_caps", "gemma_avg", "gemma_p95", "gemma_max", "gemma_pct_gt64",
                    "xlmr_avg", "xlmr_p95", "xlmr_max", "xlmr_pct_gt64"])
        for L in all_langs:
            g, x = np.array(gemma_lens[L]), np.array(xlmr_lens[L])
            w.writerow([L, len(g), round(g.mean(), 1), int(np.percentile(g, 95)), int(g.max()),
                        round(100 * (g > 64).mean(), 1), round(x.mean(), 1), int(np.percentile(x, 95)),
                        int(x.max()), round(100 * (x > 64).mean(), 1)])
    with open(PAPER / "multiling_lenstrat.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "encoder", "bucket", "n", "R10"])
        for L in all_langs:
            for which in ("float", "server"):
                for bk in ("all", "le64", "gt64"):
                    r10, n = lenstrat[(which, L)][bk]
                    w.writerow([L, which, bk, n, r10])
    with open(PAPER / "multiling_maxlen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "encoder", "maxlen", "R10"])
        for spec in OFFLINE:
            for L in all_langs:
                for ml in MAXLENS:
                    if (spec["label"], L, ml) in sweep:
                        w.writerow([L, spec["label"], ml, sweep[(spec["label"], L, ml)]])
    print("[mx] RESULT_JSON " + json.dumps({
        "toklen": {L: {"gemma_avg": round(float(np.mean(gemma_lens[L])), 1),
                       "gemma_pct_gt64": round(100 * float(np.mean(np.array(gemma_lens[L]) > 64)), 1),
                       "xlmr_pct_gt64": round(100 * float(np.mean(np.array(xlmr_lens[L]) > 64)), 1)} for L in all_langs},
        "sweep": {f"{lab}|{L}|{ml}": sweep[(lab, L, ml)] for (lab, L, ml) in sweep},
        "lenstrat": {f"{w}|{L}": lenstrat[(w, L)] for (w, L) in lenstrat},
    }, ensure_ascii=False), flush=True)
    print("[mx] DONE", flush=True)


if __name__ == "__main__":
    main()
