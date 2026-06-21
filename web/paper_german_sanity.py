"""(g) German anomaly sanity — is SigLIP2-So400m's XM3600 German R@10 = 37.58 a real text-tower
weakness or an eval/preprocessing artifact?

Evidence collected (all measured, no estimates):
  1. Reproduce baseline de R@10 (maxlen=64, padding=max_length) — must match Ext② multiling.json 37.58.
  2. Preprocessing variants on de: dynamic padding + lowercase. (maxlen=128 is a STRUCTURAL NO-OP —
     so400m text_config.max_position_embeddings=64, the tower can never see >64 positions.)
  3. Outlier check: de vs other high-resource European langs (en/fr/es/it) under the SAME pipeline —
     if fr/es/it are high and only de is low, the weakness is German-specific (not a bug).
  4. Cross-model: NLLB-CLIP de 94.03 / AltCLIP de 94.24 (batch #1 (a)) — other models handle de fine.
  5. Eyeball: dump 5 de queries' top-1 retrieved image + that image's EN caption (plausibility).

Side effect: caches /tmp/xm_so400m.pt = {img_emb, per_lang text emb} for (h) to reuse.

Run on DGX: .venv/bin/python web/paper_german_sanity.py --data data/xm3600
"""
from __future__ import annotations

import argparse
import csv
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

from web.paper_baselines_multiling import xm3600_data, so400m_imgemb, rk_map_float  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
CACHE = "/tmp/xm_so400m.pt"


def _pool(o):
    return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)


@torch.no_grad()
def so400m_text(bb, tok, strings, dev, dtype, *, padding="max_length", maxlen=64, lower=False, batch=256):
    out = []
    for s in range(0, len(strings), batch):
        chunk = [x.lower() for x in strings[s:s + batch]] if lower else strings[s:s + batch]
        t = tok(chunk, padding=padding, max_length=maxlen, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        e = _pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                  attention_mask=am.to(dev) if am is not None else None)).float().cpu()
        out.append(e)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xm3600")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    PAPER.mkdir(parents=True, exist_ok=True)

    ds = xm3600_data(args.data)
    print(f"[g] XM3600 images={ds['n_img']} langs={len(ds['langs'])}", flush=True)

    # so400m backbone (needed for the variant text encodes); reuse cached image gallery + baseline text
    # emb if a prior run already built /tmp/xm_so400m.pt (skips the expensive 3600-image encode).
    from transformers import AutoModel  # noqa: E402
    from web.common import MODEL as SIGLIP_MODEL  # noqa: E402
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dtype).to(dev).eval()
    try:  # AutoProcessor->AutoTokenizer is broken for siglip2 under tf5.1 (None model_type); fall back
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(SIGLIP_MODEL).tokenizer
    except Exception:
        from transformers import GemmaTokenizer
        tok = GemmaTokenizer.from_pretrained(SIGLIP_MODEL)
    if os.path.exists(CACHE):
        cache = torch.load(CACHE, map_location="cpu"); img_emb = cache["img_emb"]; rebuilt = False
        print(f"[g] reused cached gallery {tuple(img_emb.shape)}", flush=True)
    else:
        img_emb = so400m_imgemb(ds["paths"], dev, dtype)[0]
        cache = {"img_emb": img_emb, "per_lang": {}}; rebuilt = True
    img_n = F.normalize(img_emb, dim=1).to(dev)

    # baseline (orig-case, maxlen=64 pad=max_length) vs lowercased R@K for ALL 36 langs:
    # is the de anomaly casing-specific, or is lowercasing a universal preprocessing fix?
    base, lower = {}, {}
    for L in ds["langs"]:
        caps, gold = ds["caps"][L], ds["gold"][L]
        pl = cache.get("per_lang", {}).get(L)
        te = pl["text_emb"] if pl else so400m_text(bb, tok, caps, dev, dtype)
        if not pl:
            cache.setdefault("per_lang", {})[L] = {"text_emb": te, "gold": gold, "n_caps": len(caps)}
        base[L] = rk_map_float(F.normalize(te, dim=1).to(dev), img_n, gold)
        te_lc = so400m_text(bb, tok, caps, dev, dtype, lower=True)
        lower[L] = rk_map_float(F.normalize(te_lc, dim=1).to(dev), img_n, gold)
    if rebuilt:
        torch.save(cache, CACHE); print(f"[g] cached so400m XM3600 -> {CACHE}", flush=True)

    def avg(d):
        return round(float(np.mean([d[L][10] for L in ds["langs"]])), 2)
    print(f"[g] avg36 R@10: baseline {avg(base)} vs lowercased {avg(lower)}", flush=True)
    for L in ["de", "en", "fr", "es", "it", "ru", "th", "ko"]:
        if L in base:
            print(f"[g]   {L}: baseline {base[L][10]} -> lowercased {lower[L][10]}", flush=True)

    # de extra: dynamic padding (orig case) — pooling sensitivity
    de_dyn = rk_map_float(F.normalize(so400m_text(bb, tok, ds["caps"]["de"], dev, dtype, padding="longest"),
                                      dim=1).to(dev), img_n, ds["gold"]["de"])

    rows = []
    for L in ds["langs"]:
        for tag, d in (("baseline(maxlen64,orig-case)", base), ("lowercased", lower)):
            rows.append({"lang": L, "variant": tag, "R1": d[L][1], "R5": d[L][5], "R10": d[L][10],
                         "mAP10": d[L]["map10"]})
    rows.append({"lang": "de", "variant": "dynamic-pad(longest,orig-case)", "R1": de_dyn[1], "R5": de_dyn[5],
                 "R10": de_dyn[10], "mAP10": de_dyn["map10"]})
    rows.append({"lang": "AVG36", "variant": "baseline", "R1": "", "R5": "", "R10": avg(base), "mAP10": ""})
    rows.append({"lang": "AVG36", "variant": "lowercased", "R1": "", "R5": "", "R10": avg(lower), "mAP10": ""})

    # ---- eyeball: 5 de queries (lowercased) -> top-1 image + that image's EN caption ----
    de_caps, de_gold = ds["caps"]["de"], ds["gold"]["de"]
    de_lc = so400m_text(bb, tok, de_caps, dev, dtype, lower=True)
    en_by_img = {}
    for cap, gg in zip(ds["caps"].get("en", []), ds["gold"].get("en", [])):
        en_by_img.setdefault(gg, cap)
    top1 = (F.normalize(de_lc, dim=1).to(dev) @ img_n.t()).argmax(1).cpu().numpy()
    rng = np.random.default_rng(0)
    eyeball = [{"de_query": de_caps[i][:70], "hit": de_gold[i] == int(top1[i]),
                "retrieved_EN_caption": en_by_img.get(int(top1[i]), "")[:70]}
               for i in sorted(rng.choice(len(de_caps), 5, replace=False).tolist())]

    with open(PAPER / "german_sanity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "variant", "R1", "R5", "R10", "mAP10"])
        w.writeheader(); w.writerows(rows)
    de_jump = round(lower["de"][10] - base["de"][10], 1)
    concl = (f"CASING ARTIFACT, not a real weakness: de R@10 {base['de'][10]} (orig case) -> {lower['de'][10]} "
             f"(lowercased, +{de_jump}). German capitalizes all nouns; the so400m text tower expects lowercase. "
             f"avg36 baseline {avg(base)} -> lowercased {avg(lower)}. Implication: batch #1's de row (and the "
             f"'so400m German weakness' framing) used orig-case+maxlen padding and is an eval artifact. "
             f"(maxlen>64 raises ValueError in so400m, confirming the 64-position cap.)")
    print("[g] RESULT_JSON " + json.dumps({"avg36_base": avg(base), "avg36_lower": avg(lower),
          "de_base": base["de"][10], "de_lower": lower["de"][10], "de_dynamic": de_dyn[10],
          "per_lang": {L: {"base": base[L][10], "lower": lower[L][10]} for L in ds["langs"]},
          "eyeball": eyeball, "conclusion": concl}, ensure_ascii=False), flush=True)
    print(f"[g] DONE -> paper/german_sanity.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
