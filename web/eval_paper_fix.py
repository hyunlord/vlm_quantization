"""Paper-eval corrections (E, F) — eval-only, reuses ft113 head + cached embeddings.
Continues web/eval_paper.py on the eval_korean 5K protocol (gallery=img_h(te_img), gold caption_i->image_i).

E. TRUE binarization ceiling = each head's PRE-SIGN continuous output
   `F.normalize(BN(hash_head(emb)[:,:1024]), p=2, dim=1)` (the exact tensor SignSTE binarizes;
   NO tanh, NO sign) -> cosine Top-K. Both image & text. Must be >= 1-bit server for BOTH langs.
   Rewrites paper/upper_bound.csv: relabel float row -> backbone_float_no_head, add
   head_continuous_ceiling, keep bit1024_server.

F. text_tower bf16 on FULL 5K (was n=1000): re-encode 5K EN+KO with the bf16 so400m tower
   (enc's loaded tower), fp32 head, fp32 img_h gallery -> R@{1,5,10} + bitflip/1024.
   Updates the text_tower bf16 row in paper/precision.csv.

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/eval_paper_fix.py
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

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BITS, Encoder, pack_bits  # noqa: E402
from web.eval_paper import faiss_bin, head_codes, nrm, pair_bitflip, recall_ks  # noqa: E402

PAPER = Path(REPO) / "paper"
_ID = re.compile(r"_0*(\d+)\.jpg")
KS = (1, 5, 10)


def head_continuous(head, emb_np, enc, bit=CODE_BITS):
    """Pre-sign continuous: F.normalize(BN(hash_head(prep(emb))[:, :bit])). No tanh/sign."""
    et = enc._prep(torch.from_numpy(np.ascontiguousarray(emb_np, dtype=np.float32)).to(enc.device))
    bn = head.batch_norms[[int(b) for b in enc.bit_list].index(bit)]
    with torch.no_grad():
        raw = head.hash_head(et)[:, :bit]
        normalized = torch.nn.functional.normalize(bn(raw), p=2, dim=1)  # == what SignSTE signs
    return normalized.float().cpu().numpy()  # already L2-normalized


def cos_recall(txt_c, img_c, n, ks=KS):
    sims = txt_c @ img_c.T  # both L2-normalized -> cosine
    return {k: round(100 * np.mean([i in np.argpartition(-sims[i], k)[:k] for i in range(n)]), 2) for k in ks}


def main():
    enc = Encoder()
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy()
    te_en = EC["test"]["txt"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"])
        kf[int(m.group(1)) if m else -1] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    n = len(te_ids)
    gold = list(range(n))
    TXT = {"EN": te_en, "KO": ko_emb}

    # anchor
    gal1024 = pack_bits(head_codes(enc.img_h, te_img, enc, CODE_BITS))
    ix1024 = faiss_bin(gal1024, CODE_BITS)
    anc = {L: recall_ks(ix1024, pack_bits(head_codes(enc.txt_h, TXT[L], enc, CODE_BITS)), gold) for L in ("EN", "KO")}
    print(f"[fix] ANCHOR 1024 server R@10 EN {anc['EN'][10]} / KO {anc['KO'][10]} (expect 79.92/71.08)", flush=True)

    # ===== E. head-continuous ceiling =====
    img_c = head_continuous(enc.img_h, te_img, enc)
    hc = {L: cos_recall(head_continuous(enc.txt_h, TXT[L], enc), img_c, n) for L in ("EN", "KO")}
    bb = {L: cos_recall(nrm(TXT[L]), nrm(te_img), n) for L in ("EN", "KO")}  # backbone float (no head)
    ok = all(hc[L][10] >= anc[L][10] for L in ("EN", "KO"))
    print(f"[fix] E head-continuous ceiling: EN {hc['EN']} / KO {hc['KO']} | "
          f">=1bit both: {ok}", flush=True)
    print(f"[fix] E backbone-float(no head): EN R@10 {bb['EN'][10]} / KO {bb['KO'][10]}", flush=True)
    with open(PAPER / "upper_bound.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "EN_R1", "EN_R5", "EN_R10", "KO_R1", "KO_R5", "KO_R10"])
        w.writerow(["head_continuous_ceiling", hc["EN"][1], hc["EN"][5], hc["EN"][10], hc["KO"][1], hc["KO"][5], hc["KO"][10]])
        w.writerow(["bit1024_server", anc["EN"][1], anc["EN"][5], anc["EN"][10], anc["KO"][1], anc["KO"][5], anc["KO"][10]])
        w.writerow(["backbone_float_no_head", bb["EN"][1], bb["EN"][5], bb["EN"][10], bb["KO"][1], bb["KO"][5], bb["KO"][10]])

    # ===== F. text_tower bf16 FULL 5K =====
    enc._load_tower()  # bf16 cuda tower (same as the demo path)
    tok, tower = enc._tok, enc._tower

    def bf16_embed(strings, batch=256):
        out = []
        for s in range(0, len(strings), batch):
            t = tok(strings[s:s + batch], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            am = t.get("attention_mask")
            with torch.no_grad():
                e = enc._pool(tower.text_model(input_ids=t["input_ids"].to(enc.device),
                              attention_mask=am.to(enc.device) if am is not None else None)).float()
            out.append(e.cpu().numpy())
        return np.concatenate(out, 0)

    flips, R = [], {}
    for L, caps in [("EN", en_caps), ("KO", ko_caps)]:
        bf = bf16_embed(caps)
        bf_code = pack_bits(head_codes(enc.txt_h, bf, enc, CODE_BITS))
        fp_code = pack_bits(head_codes(enc.txt_h, TXT[L], enc, CODE_BITS))  # fp32-cache ref
        flips.append(pair_bitflip(bf_code, fp_code))
        R[L] = recall_ks(ix1024, bf_code, gold)
    flip = round(float(np.mean(flips)), 2)
    print(f"[fix] F text_tower bf16 FULL 5K: flip {flip}/1024 | EN {R['EN']} / KO {R['KO']}", flush=True)

    ppath = PAPER / "precision.csv"
    rows = list(csv.DictReader(open(ppath)))
    for r in rows:
        if r["target"] == "text_tower" and r["dtype"] == "bf16":
            r["bitflips_per_1024"] = str(flip)
            r["EN_R10"] = str(R["EN"][10]); r["KO_R10"] = str(R["KO"][10])
            r["note"] = "so400m tower bf16(cuda) vs fp32 cache, head fp32, FULL 5K"
    with open(ppath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    print("[fix] RESULT_JSON " + json.dumps({
        "anchor": {"EN": anc["EN"][10], "KO": anc["KO"][10]},
        "head_continuous": {"EN": hc["EN"], "KO": hc["KO"]},
        "backbone_float": {"EN": bb["EN"][10], "KO": bb["KO"][10]},
        "ceiling_ge_1bit_both": ok,
        "tower_bf16_full5k": {"flip": flip, "EN_R10": R["EN"][10], "KO_R10": R["KO"][10]}}), flush=True)
    print("[fix] DONE -> paper/upper_bound.csv, paper/precision.csv", flush=True)


if __name__ == "__main__":
    main()
