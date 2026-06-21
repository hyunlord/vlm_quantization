"""(i)+(iii)+(v) so400m text lowercasing correction on COCO 5K.

Batch #2 (g) found the so400m (SigLIP2) text tower expects LOWERCASE input (official SiglipTokenizer has
do_lower_case=True + canonicalize_text), but the repo tokenizes via the case-sensitive Gemma tokenizer
(self.processor.tokenizer) and never lowercases -> cased-script text scores are suppressed. This re-derives
all COCO so400m-text numbers with the fix = `.lower()` + padding=max_length/maxlen=64 (the verified anchor;
dynamic padding is worse, maxlen>64 errors).

(i) AUDIT: confirm emb_cache test-txt is orig-case (cos vs orig ~1.0, vs lower <1) -> the 79.92 anchor used
    orig case; report corrected COCO EN (KO unchanged, Korean is caseless).
(iii) so400m COCO rows orig vs lowercased: naive-sign (no head), float (cosine ceiling), server (ft113 1bit).
(v) offline E5/MiniLM: native vs lowercased (model-specific — these are NOT so400m; keep native). Sanity only.

Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_lowercase_fix.py
"""
from __future__ import annotations

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
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
_ID = re.compile(r"_0*(\d+)\.jpg")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def r10_map(ix, q, gold):
    _, I = ix.search(q, 10)
    r = round(100 * np.mean([gold[i] in I[i, :10] for i in range(len(gold))]), 2)
    ap = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0) for i in range(len(gold))]
    return r, round(100 * float(np.mean(ap)), 2)


def float_r10_map(txt_n, img_n, gold):
    idx = (txt_n @ img_n.t()).topk(10, dim=1).indices.cpu().numpy()
    r = round(100 * np.mean([gold[i] in idx[i, :10] for i in range(len(gold))]), 2)
    ap = [(1.0 / (np.where(idx[i] == gold[i])[0][0] + 1) if gold[i] in idx[i] else 0.0) for i in range(len(gold))]
    return r, round(100 * float(np.mean(ap)), 2)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"])
        if m:
            kf[int(m.group(1))] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    te_img = EC["test"]["img"].float().numpy()
    n = len(te_ids); gold = list(range(n))

    enc = Encoder()
    from transformers import AutoModel, GemmaTokenizer
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    tok = GemmaTokenizer.from_pretrained(SIGLIP_MODEL)

    @torch.no_grad()
    def so400m_txt(strings, lower):
        out = []
        for s in range(0, len(strings), 256):
            chunk = [x.lower() for x in strings[s:s + 256]] if lower else strings[s:s + 256]
            t = tok(chunk, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            am = t.get("attention_mask")
            o = bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)
            e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
            out.append(e.float().cpu())
        return torch.cat(out)

    # ---- (i) audit: cached test-txt vs orig/lower ----
    cached = F.normalize(EC["test"]["txt"][:64].float(), dim=1)
    cos_orig = float((cached * F.normalize(so400m_txt(en_caps[:64], False), dim=1)).sum(1).mean())
    cos_low = float((cached * F.normalize(so400m_txt(en_caps[:64], True), dim=1)).sum(1).mean())
    print(f"[lc] AUDIT cache vs orig-case cos {cos_orig:.4f} | vs lowercased cos {cos_low:.4f} "
          f"-> cache is {'ORIG-CASE (anchor used orig case)' if cos_orig > cos_low else 'lowercased'}", flush=True)

    rows = []
    img_n = F.normalize(torch.from_numpy(te_img), dim=1)
    img_n_dev = img_n.to(dev)
    # server gallery (ft113 1bit) + naive gallery (sign raw img) reused across variants
    gal_server = enc.image_codes_packed(te_img); ix_server = faiss_bin(gal_server, enc.bits)
    gal_naive = pack_bits(img_n.numpy()); ix_naive = faiss_bin(gal_naive, te_img.shape[1])

    for lang, caps, en_is_ko in (("EN", en_caps, False), ("KO", ko_caps, True)):
        for variant in ("orig", "lower"):
            lower = variant == "lower"
            te = so400m_txt(caps, lower)
            te_n = F.normalize(te, dim=1).to(dev)
            # server (ft113 1bit)
            sr, sm = r10_map(ix_server, pack_bits(enc._codes_pm1(enc.txt_h, te)), gold)
            rows.append({"path": "server(ft113 1bit)", "model": "SigLIP2-so400m", "variant": variant,
                         "lang": lang, "R10": sr, "mAP10": sm})
            # float ceiling
            fr, fm = float_r10_map(te_n, img_n_dev, gold)
            rows.append({"path": "float(cosine)", "model": "SigLIP2-so400m", "variant": variant,
                         "lang": lang, "R10": fr, "mAP10": fm})
            # naive-sign (no head)
            nr, nm = r10_map(ix_naive, pack_bits(te_n.cpu().numpy()), gold)
            rows.append({"path": "naive-sign(no head)", "model": "SigLIP2-so400m", "variant": variant,
                         "lang": lang, "R10": nr, "mAP10": nm})
            print(f"[lc] {lang} {variant}: server {sr} | float {fr} | naive {nr}", flush=True)

    # ---- (v) offline E5/MiniLM: native vs lowercased (NOT so400m; expect ~no gain -> keep native) ----
    @torch.no_grad()
    def offline_codes(head_path, caps, lower):
        ck = torch.load(head_path, map_location="cpu")
        student, prefix = ck["student"], (ck.get("prefix") or "")
        bits = [int(b) for b in ck["bits"]]; obi = bits.index(enc.bits)
        th = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0); th.load_state_dict(ck["txt_h"]); th.to(dev).eval()
        from transformers import AutoModel as AM, AutoTokenizer
        m = AM.from_pretrained(student).to(dev).eval(); t2 = AutoTokenizer.from_pretrained(student)
        out = []
        for s in range(0, len(caps), 256):
            chunk = [x.lower() for x in caps[s:s + 256]] if lower else caps[s:s + 256]
            tt = t2([prefix + x for x in chunk], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            o = m(input_ids=tt["input_ids"].to(dev), attention_mask=tt["attention_mask"].to(dev)).last_hidden_state
            msk = tt["attention_mask"].to(dev).unsqueeze(-1).float()
            e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            out.append(F.normalize(e, dim=1).float())
        del m
        with torch.no_grad():
            return pack_bits(th(torch.cat(out).to(dev))[obi]["binary"].detach().cpu().numpy())
    off_rows = []
    for label, hp in (("e5-small", "/tmp/txt_h_e5.pt"),
                      ("MiniLM", "/tmp/txt_h_paraphrase-multilingual-MiniLM-L12-v2.pt")):
        if not os.path.exists(hp):
            continue
        for lang, caps in (("EN", en_caps), ("KO", ko_caps)):
            for variant in ("native", "lower"):
                r, mp = r10_map(ix_server, offline_codes(hp, caps, variant == "lower"), gold)
                off_rows.append({"path": f"offline {label}", "model": label, "variant": variant,
                                 "lang": lang, "R10": r, "mAP10": mp})
                print(f"[lc] offline {label} {lang} {variant}: R@10 {r}", flush=True)

    with open(PAPER / "baseline_lc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "model", "variant", "lang", "R10", "mAP10"])
        w.writeheader(); w.writerows(rows + off_rows)
    print("[lc] RESULT_JSON " + json.dumps({"audit": {"cos_orig": round(cos_orig, 4), "cos_lower": round(cos_low, 4)},
          "rows": rows, "offline": off_rows}), flush=True)
    print(f"[lc] DONE -> paper/baseline_lc.csv ({len(rows)+len(off_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
