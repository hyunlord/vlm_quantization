"""(1-core) COCO main/baseline/encoders so400m-text rows, lowercased, FULL R@{1,5,10}+mAP@10.

Batch #3 only gave R@10/mAP@10 for the COCO so400m-text paths. The paper's main/baseline/encoder
tables also need R@1/R@5. This recomputes every COCO so400m-text path (server=ft113 1bit, float=raw
backbone cosine ceiling [+ head-continuous], naive=sign(raw emb)) for EN+KO, orig-case vs lowercased,
at R@{1,5,10}+mAP@10.

ALSO builds the lowercased caches reused (no backbone reload) by precision / bits / multiling:
  /tmp/coco_en_lc.pt    = lowercased COCO test EN so400m text emb (bf16 backbone -> float; the pipeline dtype)
  /tmp/coco_en_orig.pt  = orig-case COCO test EN re-encode (SAME bb) for a controlled orig-vs-lower delta
  /tmp/coco_en_lc_fp32.pt / _orig_fp32.pt = fp32-backbone encodes (for the precision text_tower row)
  /tmp/xm_so400m_lc.pt  = {img_emb(copy of orig cache), per_lang:{L:{text_emb(lower), gold}}}

Model-specific rule: ONLY so400m text is lowercased. KO is caseless (.lower() is a string no-op) -> orig==lower
(asserted/measured). Anchors: server EN lower R@10 = 81.24 (baseline_lc.csv); XM de lower ~96.5, avg36 ~74.5
(german_sanity.csv). orig server EN R@10 ~79.98, R@1 ~41.6 (encoders.csv).

Run: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_coco_lc_full.py --data data/xm3600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import MODEL as SIGLIP_MODEL, Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
_ID = re.compile(r"_0*(\d+)\.jpg")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _pool(o):
    return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)


def build_tok():
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(SIGLIP_MODEL).tokenizer
    except Exception:
        from transformers import GemmaTokenizer
        return GemmaTokenizer.from_pretrained(SIGLIP_MODEL)


@torch.no_grad()
def so400m_text(bb, tok, strings, dt, *, lower=False, batch=256):
    out = []
    for s in range(0, len(strings), batch):
        chunk = [x.lower() for x in strings[s:s + batch]] if lower else strings[s:s + batch]
        t = tok(chunk, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        e = _pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                  attention_mask=am.to(dev) if am is not None else None)).float().cpu()
        out.append(e)
    return torch.cat(out)


def xm3600_data(data_dir):
    data = Path(data_dir)
    capf = data / "captions.jsonl"
    if not capf.exists():
        capf = next(data.rglob("captions.jsonl"))
    imgmap = {p.stem: p for p in (data / "images").rglob("*.jpg")} if (data / "images").exists() \
        else {p.stem: p for p in data.rglob("*.jpg")}
    recs = [json.loads(l) for l in open(capf, encoding="utf-8") if l.strip()]
    paths, key2idx = [], {}
    for rec in recs:
        p = imgmap.get(rec.get("image/key"))
        if p is not None:
            key2idx[rec["image/key"]] = len(paths); paths.append(p)
    caps, gold = {}, {}
    for rec in recs:
        idx = key2idx.get(rec.get("image/key"))
        if idx is None:
            continue
        for lang, val in rec.items():
            if not isinstance(val, dict):
                continue
            for cap in (val.get("caption") or []):
                caps.setdefault(lang, []).append(cap); gold.setdefault(lang, []).append(idx)
    return dict(paths=paths, caps=caps, gold=gold, langs=sorted(caps.keys()), n_img=len(paths))


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def rk_bin(ix, q_packed, gold):
    _, I = ix.search(np.ascontiguousarray(q_packed), 10)
    out = {k: round(100 * float(np.mean([gold[i] in I[i, :k] for i in range(len(gold))])), 2) for k in KS}
    ap = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0)
          for i in range(len(gold))]
    out["map10"] = round(100 * float(np.mean(ap)), 2)
    return out


def rk_float(txt_n, img_n, gold):
    idx = (txt_n @ img_n.t()).topk(10, dim=1).indices.cpu().numpy()
    out = {k: round(100 * float(np.mean([gold[i] in idx[i, :k] for i in range(len(gold))])), 2) for k in KS}
    ap = [(1.0 / (np.where(idx[i] == gold[i])[0][0] + 1) if gold[i] in idx[i] else 0.0)
          for i in range(len(gold))]
    out["map10"] = round(100 * float(np.mean(ap)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xm3600")
    args = ap.parse_args()
    PAPER.mkdir(parents=True, exist_ok=True)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32

    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en_caps = [str(c) for c in EC["test"]["captions"]]
    te_img = EC["test"]["img"].float().numpy()
    n = len(te_ids); gold = list(range(n))

    # KO test caption strings (caseless; for the orig==lower invariance spot-check)
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); m = _ID.search(e.get("image_path", ""))
        if m:
            kf[int(m.group(1))] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    ko_caseless = all(c == c.lower() for c in ko_caps)

    # ---- backbone (bf16 on cuda, like the existing caches) ----
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dt).to(dev).eval()
    tok = build_tok()

    # encode COCO EN orig + lower (bf16->float, the pipeline dtype); KO orig + lower
    en_orig = so400m_text(bb, tok, en_caps, dt, lower=False)
    en_low = so400m_text(bb, tok, en_caps, dt, lower=True)
    ko_orig = so400m_text(bb, tok, ko_caps, dt, lower=False)
    ko_low = so400m_text(bb, tok, ko_caps, dt, lower=True)
    torch.save(en_low, "/tmp/coco_en_lc.pt")
    torch.save(en_orig, "/tmp/coco_en_orig.pt")
    print(f"[core] KO caseless(strings)={ko_caseless} | KO orig-vs-lower emb cos="
          f"{float(F.cosine_similarity(ko_orig, ko_low).mean()):.4f}", flush=True)

    # fp32-backbone encodes for the precision text_tower(bf16-vs-fp32) row
    try:
        bb32 = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float32).to(dev).eval()
        en_low_fp32 = so400m_text(bb32, tok, en_caps, torch.float32, lower=True)
        en_orig_fp32 = so400m_text(bb32, tok, en_caps, torch.float32, lower=False)
        torch.save(en_low_fp32, "/tmp/coco_en_lc_fp32.pt")
        torch.save(en_orig_fp32, "/tmp/coco_en_orig_fp32.pt")
        del bb32
        if dev == "cuda":
            torch.cuda.empty_cache()
        print("[core] cached fp32-backbone COCO EN encodes (for precision text_tower row)", flush=True)
    except Exception as ex:
        print(f"[core] WARN fp32 backbone encode failed ({ex}); precision text_tower row will be flagged", flush=True)

    # ---- evaluators ----
    enc = Encoder()
    img_n = F.normalize(torch.from_numpy(te_img), dim=1)
    img_n_dev = img_n.to(dev)
    gal_server = enc.image_codes_packed(te_img); ix_server = faiss_bin(gal_server, enc.bits)
    gal_naive = pack_bits(img_n.numpy()); ix_naive = faiss_bin(gal_naive, te_img.shape[1])

    # head-continuous ceiling gallery/query helper (tanh-normalized continuous codes, cosine)
    @torch.no_grad()
    def head_cont(head, emb_t):
        outs = head(enc._prep(emb_t.to(dev)))
        return F.normalize(outs[enc.bit_index]["continuous"], dim=1)
    img_cont = head_cont(enc.img_h, torch.from_numpy(te_img))

    rows = []

    def emit(table, row, lang, variant, te_emb):
        te_n = F.normalize(te_emb, dim=1).to(dev)
        recs = {}
        recs["server"] = rk_bin(ix_server, pack_bits(enc._codes_pm1(enc.txt_h, te_emb)), gold)
        recs["float_raw"] = rk_float(te_n, img_n_dev, gold)
        recs["naive"] = rk_bin(ix_naive, pack_bits(te_n.cpu().numpy()), gold)
        recs["float_headcont"] = rk_float(head_cont(enc.txt_h, te_emb), img_cont, gold)
        for path, r in recs.items():
            rows.append({"table": table, "row": f"{row}|{path}", "lang": lang, "R1": r[1], "R5": r[5],
                         "R10": r[10], "mAP10": r["map10"], "preproc": variant})
        return recs

    for lang, e_orig, e_low in (("EN", en_orig, en_low), ("KO", ko_orig, ko_low)):
        ro = emit("coco", "so400m", lang, "orig", e_orig)
        rl = emit("coco", "so400m", lang, "lower", e_low)
        print(f"[core] {lang}: server R@10 {ro['server'][10]}->{rl['server'][10]} | "
              f"float_raw {ro['float_raw'][10]}->{rl['float_raw'][10]} | naive {ro['naive'][10]}->{rl['naive'][10]} "
              f"| server R@1 {ro['server'][1]}->{rl['server'][1]}", flush=True)

    # ---- XM3600 lowercased cache for (2)/(3): per-lang lowercased text emb + reuse orig img gallery ----
    XMc = torch.load("/tmp/xm_so400m.pt", map_location="cpu")
    ds = xm3600_data(args.data)
    per_lang = {}
    de_chk = avg_chk = None
    img_xn = F.normalize(XMc["img_emb"], dim=1).to(dev)
    r10s = []
    for L in ds["langs"]:
        caps, g = ds["caps"][L], ds["gold"][L]
        te_lc = so400m_text(bb, tok, caps, dt, lower=True)
        per_lang[L] = {"text_emb": te_lc, "gold": g}
        rr = rk_float(F.normalize(te_lc, dim=1).to(dev), img_xn, g)
        r10s.append(rr[10])
        if L == "de":
            de_chk = rr[10]
    avg_chk = round(float(np.mean(r10s)), 2)
    torch.save({"img_emb": XMc["img_emb"], "per_lang": per_lang}, "/tmp/xm_so400m_lc.pt")
    print(f"[core] XM3600 lowercased cache built: de R@10(float)={de_chk} (anchor 96.54) | "
          f"avg36 R@10(float)={avg_chk} (anchor 74.46)", flush=True)

    cols = ["table", "row", "lang", "R1", "R5", "R10", "mAP10", "preproc"]
    with open(PAPER / "coco_lc_full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[core] RESULT_JSON " + json.dumps({"rows": rows, "ko_caseless": ko_caseless,
          "xm_de_float_lower": de_chk, "xm_avg36_float_lower": avg_chk}, ensure_ascii=False), flush=True)
    print(f"[core] DONE -> paper/coco_lc_full.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
