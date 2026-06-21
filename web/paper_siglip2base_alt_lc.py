"""(a1) Remaining lowercasing gap — SigLIP2-base backbone row (tab:backbones) + AltCLIP native check.

Batch #4 corrected the so400m(ft113) row; SigLIP2-base is the SAME SiglipTokenizer family (do_lower_case=True)
so its backbone-table row is still orig-case. Re-derive its server 1-bit (EN/KO R@10+R1), head-continuous
ceiling (EN/KO), and the precision flips (head/emb bf16/int8) with the so400m-family TEXT lowercased, reusing
the cached bb_siglip2-base head + image/test embeddings (only text re-encoded). KO is caseless -> reuse cached
ko emb (unchanged). MiniLM head-adapt = offline, native -> unchanged (value_lower == value_orig).

AltCLIP-m18 (XLM-R text, case-SENSITIVE) is NOT SigLIP-family: do a 1-shot EN orig-vs-lowercased cosine R@10
(reusing cached AltCLIP test image emb) to confirm lowercasing does NOT help -> keep native (documented).

Writes paper/encoders_baselines_lc.csv (header + backbone rows); paper_multiling_baseline_lc.py appends to it.
value_orig is read from paper/backbones.csv. Flip scheme mirrors eval_paper.py/backbone_run.py §C exactly.

Run: .venv/bin/python web/paper_siglip2base_alt_lc.py
"""
from __future__ import annotations

import copy
import csv
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
from web.paper_backbone_run import load_altclip, emb_texts_altclip  # noqa: E402

PAPER = Path(REPO) / "paper"
OUT = PAPER / "encoders_baselines_lc.csv"
COLS = ["table", "backbone", "lang_or_metric", "value_orig", "value_lower", "preproc"]
KS = (1, 5, 10)
SIGLIP_BASE = "google/siglip2-base-patch16-256"
dev = "cuda" if torch.cuda.is_available() else "cpu"
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pack_bits(c):
    b = (np.asarray(c) > 0).astype(np.uint8)
    if b.ndim == 1:
        b = b[None, :]
    return np.ascontiguousarray(np.packbits(b, axis=1, bitorder="big"), dtype=np.uint8)


def faiss_bin(p, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(p)); return ix


def recall_ks(ix, q, gold):
    _, I = ix.search(np.ascontiguousarray(q), max(KS))
    return {k: round(100 * float(np.mean([gold[i] in I[i, :k] for i in range(len(gold))])), 2) for k in KS}


def cosine_recall(txt, img, gold):
    tn = F.normalize(torch.from_numpy(txt), dim=1).to(dev)
    ig = F.normalize(torch.from_numpy(img), dim=1).to(dev)
    idx = (tn @ ig.t()).topk(max(KS), dim=1).indices.cpu().numpy()
    return {k: round(100 * float(np.mean([gold[i] in idx[i, :k] for i in range(len(gold))])), 2) for k in KS}


def pair_bitflip(a, b):
    return float(_LUT[np.bitwise_xor(a, b)].sum(1).mean())


def cast_emb_np(x, dt):
    if dt == "int8":
        s = np.abs(x).max(axis=1, keepdims=True) / 127.0 + 1e-9
        return (np.round(x / s).clip(-127, 127) * s).astype(np.float32)
    if dt == "bf16":
        return torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    return x.astype(np.float16).astype(np.float32)


def build_tok(model_id):
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_id).tokenizer
    except Exception:
        from transformers import GemmaTokenizer
        return GemmaTokenizer.from_pretrained(model_id)


@torch.no_grad()
def siglip_text(bb, tok, strings, lower, batch=256):
    out = []
    for s in range(0, len(strings), batch):
        chunk = [x.lower() for x in strings[s:s + batch]] if lower else strings[s:s + batch]
        t = tok(chunk, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        o = bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)
        e = (o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1))
        out.append(e.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def head_bin(head, emb, bi):
    return head(F.normalize(emb.to(dev), dim=1))[bi]["binary"].detach().cpu().numpy()


@torch.no_grad()
def head_cont(head, emb, bi):
    return head(F.normalize(emb.to(dev), dim=1))[bi]["continuous"].detach().cpu().numpy()


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    orig = {r["backbone"]: r for r in csv.DictReader(open(PAPER / "backbones.csv"))}
    SB = next(r for k, r in orig.items() if "siglip2-base" in k)

    ck = torch.load("/tmp/bb_siglip2-base_head.pt", map_location="cpu")
    bits = [int(b) for b in ck["bits"]]; bi = bits.index(1024)
    img_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0); img_h.load_state_dict(ck["img_h"]); img_h.to(dev).eval()
    txt_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0); txt_h.load_state_dict(ck["txt_h"]); txt_h.to(dev).eval()
    TE = torch.load("/tmp/bb_siglip2-base_test.pt", map_location="cpu")
    img, en_cap_emb_orig, ko_emb = TE["img"].float(), TE["en"].float(), TE["ko"].float()
    n = img.shape[0]; gold = list(range(n))

    # re-encode COCO test EN through SigLIP2-base text tower LOWERCASED (need the caption strings)
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    en_caps = [str(c) for c in EC["test"]["captions"]]
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(SIGLIP_BASE, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    tok = build_tok(SIGLIP_BASE)
    en_lc = siglip_text(bb, tok, en_caps, lower=True)
    en_orig = siglip_text(bb, tok, en_caps, lower=False)  # for the orig-reproduce sanity
    del bb
    if dev == "cuda":
        torch.cuda.empty_cache()

    # gallery + image continuous (image side unchanged)
    gal = pack_bits(head_bin(img_h, img, bi)); ix = faiss_bin(gal, 1024)
    img_cont = head_cont(img_h, img, bi)

    rows = []

    def emit(backbone, metric, vorig, vlower, preproc):
        rows.append({"table": "backbones", "backbone": backbone, "lang_or_metric": metric,
                     "value_orig": vorig, "value_lower": vlower, "preproc": preproc})

    # ---- SigLIP2-base server 1-bit + ceiling, lowercased EN; KO caseless (cached) ----
    en_srv_lc = recall_ks(ix, pack_bits(head_bin(txt_h, en_lc, bi)), gold)
    en_srv_or = recall_ks(ix, pack_bits(head_bin(txt_h, en_orig, bi)), gold)  # sanity vs backbones.csv 78.12/38.16
    ko_srv = recall_ks(ix, pack_bits(head_bin(txt_h, ko_emb, bi)), gold)
    en_ceil_lc = cosine_recall(head_cont(txt_h, en_lc, bi), img_cont, gold)
    ko_ceil = cosine_recall(head_cont(txt_h, ko_emb, bi), img_cont, gold)
    emit("SigLIP2-base", "server_1bit_EN_R10", SB["server_1bit_EN_R10"], en_srv_lc[10], "lowercased")
    emit("SigLIP2-base", "server_1bit_EN_R1", SB["server_1bit_EN_R1"], en_srv_lc[1], "lowercased")
    emit("SigLIP2-base", "server_1bit_KO_R10", SB["server_1bit_KO_R10"], ko_srv[10], "native(caseless)")
    emit("SigLIP2-base", "server_1bit_KO_R1", SB["server_1bit_KO_R1"], ko_srv[1], "native(caseless)")
    emit("SigLIP2-base", "head_cont_ceiling_EN_R10", SB["head_cont_ceiling_EN_R10"], en_ceil_lc[10], "lowercased")
    emit("SigLIP2-base", "head_cont_ceiling_KO_R10", SB["head_cont_ceiling_KO_R10"], ko_ceil[10], "native(caseless)")

    # ---- precision flips (EN+KO avg, lowercased EN), eval_paper §C scheme ----
    ref_en = pack_bits(head_bin(txt_h, en_lc, bi)); ref_ko = pack_bits(head_bin(txt_h, ko_emb, bi))

    def txt_flip(codes_en, codes_ko):
        return round(float(np.mean([pair_bitflip(codes_en, ref_en), pair_bitflip(codes_ko, ref_ko)])), 2)

    th_bf = copy.deepcopy(txt_h).to(dev, torch.bfloat16).eval()
    f_head_bf = txt_flip(pack_bits(th_bf(F.normalize(en_lc, dim=1).to(dev, torch.bfloat16))[bi]["binary"].detach().float().cpu().numpy()),
                         pack_bits(th_bf(F.normalize(ko_emb, dim=1).to(dev, torch.bfloat16))[bi]["binary"].detach().float().cpu().numpy()))
    import torch.ao.quantization as aoq
    th_q = aoq.quantize_dynamic(copy.deepcopy(txt_h).cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
    f_head_int8 = txt_flip(pack_bits(th_q(F.normalize(en_lc, dim=1).cpu())[bi]["binary"].detach().cpu().numpy()),
                           pack_bits(th_q(F.normalize(ko_emb, dim=1).cpu())[bi]["binary"].detach().cpu().numpy()))
    f_emb_bf = txt_flip(pack_bits(head_bin(txt_h, torch.from_numpy(cast_emb_np(en_lc.numpy(), "bf16")), bi)),
                        pack_bits(head_bin(txt_h, torch.from_numpy(cast_emb_np(ko_emb.numpy(), "bf16")), bi)))
    f_emb_int8 = txt_flip(pack_bits(head_bin(txt_h, torch.from_numpy(cast_emb_np(en_lc.numpy(), "int8")), bi)),
                          pack_bits(head_bin(txt_h, torch.from_numpy(cast_emb_np(ko_emb.numpy(), "int8")), bi)))
    emit("SigLIP2-base", "head_bf16_flip", SB["head_bf16_flip"], f_head_bf, "lowercased")
    emit("SigLIP2-base", "head_int8_flip", SB["head_int8_flip"], f_head_int8, "lowercased")
    emit("SigLIP2-base", "emb_bf16_flip", SB["emb_bf16_flip"], f_emb_bf, "lowercased")
    emit("SigLIP2-base", "emb_int8_flip", SB["emb_int8_flip"], f_emb_int8, "lowercased")
    emit("SigLIP2-base", "MiniLM_headadapt_EN_R10", SB["MiniLM_headadapt_EN_R10"], SB["MiniLM_headadapt_EN_R10"], "native(offline, unchanged)")
    emit("SigLIP2-base", "MiniLM_headadapt_KO_R10", SB["MiniLM_headadapt_KO_R10"], SB["MiniLM_headadapt_KO_R10"], "native(offline, unchanged)")
    print(f"[a1] SigLIP2-base EN server R@10 orig(reproduce {SB['server_1bit_EN_R10']}) {en_srv_or[10]} -> "
          f"lower {en_srv_lc[10]} | R@1 {en_srv_lc[1]} | ceiling {en_ceil_lc[10]} | KO {ko_srv[10]}(caseless) | "
          f"flips head bf16/int8 {f_head_bf}/{f_head_int8} emb {f_emb_bf}/{f_emb_int8}", flush=True)

    # ---- AltCLIP 1-shot: EN orig vs lowercased (cached AltCLIP image), native is case-sensitive XLM-R ----
    try:
        AT = torch.load("/tmp/bb_altclip-m18_test.pt", map_location="cpu")
        m, proc = load_altclip("BAAI/AltCLIP-m18", dev, torch.bfloat16 if dev == "cuda" else torch.float32)
        at_en_or = emb_texts_altclip(m, proc, en_caps, dev, torch.bfloat16 if dev == "cuda" else torch.float32)
        at_en_lc = emb_texts_altclip(m, proc, [c.lower() for c in en_caps], dev, torch.bfloat16 if dev == "cuda" else torch.float32)
        atimg = AT["img"].float().numpy()
        r_or = cosine_recall(at_en_or.numpy(), atimg, gold)[10]
        r_lc = cosine_recall(at_en_lc.numpy(), atimg, gold)[10]
        verdict = "keep native (lowercasing hurts/flat)" if r_lc <= r_or + 0.1 else "lowercasing helps (unexpected)"
        emit("AltCLIP-m18", "float_EN_R10 (native vs lower check)", round(r_or, 2), round(r_lc, 2),
             f"native — {verdict}")
        print(f"[a1] AltCLIP EN float R@10: native {r_or} vs lowercased {r_lc} -> {verdict}", flush=True)
    except Exception as ex:
        emit("AltCLIP-m18", "float_EN_R10 (native vs lower check)", "", "FLAGGED", f"load failed: {repr(ex)[:60]}")
        print(f"[a1] AltCLIP check FAILED: {repr(ex)[:100]}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"[a1] DONE -> paper/encoders_baselines_lc.csv ({len(rows)} backbone rows)", flush=True)


if __name__ == "__main__":
    main()
