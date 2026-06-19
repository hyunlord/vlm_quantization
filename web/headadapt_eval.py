"""C1 (cycle 5) parity eval — offline path = NATIVE e5 -> txt_h' (head adapted to e5),
vs the so400m server path, on the FROZEN image codes. Same harness as distill_eval.py
(eval_korean 5K-gallery protocol + index.bin Top-10 overlap), reported with deltas vs
cycle 4 (distilled e5 + so400m txt_h: EN R@10 70.86 / KO 62.50, overlap 0.512 / 0.451).

cos-to-so400m is meaningless here (different text space), so the code-space diagnostic is
the matched-pair text<->image average Hamming (lower = better aligned to the fixed codes).

Run on DGX (after headadapt_train.py):
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/headadapt_eval.py --txt-head /tmp/txt_h_e5.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, CODE_BYTES, Encoder, pack_bits  # noqa: E402

_ID = re.compile(r"_0*(\d+)\.jpg")
C4 = {"EN": {"R10": 70.86, "overlap": 0.512}, "KO": {"R10": 62.50, "overlap": 0.451}}
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _cocoid(p):
    m = _ID.search(p)
    return int(m.group(1)) if m else -1


def _faiss(packed):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(CODE_BITS); ix.add(packed); return ix


def _recall(gal_ix, qcodes, gold_ids, gal_ids, k=10):
    _, I = gal_ix.search(qcodes, k)
    return round(100 * np.mean([gold_ids[q] in [gal_ids[j] for j in I[q]]
                                for q in range(len(gold_ids))]), 2)


def _pair_hamming(a, b):  # mean per-row Hamming between aligned packed codes
    return float(_LUT[np.bitwise_xor(a, b)].sum(1).mean())


class E5:
    def __init__(self, name, dev):
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.torch = torch
        self.dev = dev
        self.m = AutoModel.from_pretrained(name).to(dev).eval()
        self.tok = AutoTokenizer.from_pretrained(name)

    def embed(self, strings, maxlen=64, batch=256):
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(strings), batch):
                t = self.tok(strings[s:s + batch], padding="max_length", max_length=maxlen,
                             truncation=True, return_tensors="pt")
                o = self.m(t["input_ids"].to(self.dev), t["attention_mask"].to(self.dev)).last_hidden_state
                msk = t["attention_mask"].to(self.dev).unsqueeze(-1).float()
                e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(e, dim=1).float().cpu().numpy())
        return np.concatenate(out, 0)


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--txt-head", default="/tmp/txt_h_e5.pt")
    p.add_argument("--static", default="web/static")
    p.add_argument("--n-overlap", type=int, default=1000)
    args = p.parse_args()

    enc = Encoder()  # frozen ft113 img_h/txt_h + packing
    ck = torch.load(args.txt_head, map_location="cpu")
    txth = NestedHashLayer(ck["embed"], ck["hidden"], [int(b) for b in ck["bits"]], 0.0)
    txth.load_state_dict(ck["txt_h"]); txth.to(enc.device).eval()
    bi = [int(b) for b in ck["bits"]].index(CODE_BITS)
    e5 = E5(ck["student"], enc.device)

    def offline_codes(strings):
        emb = e5.embed(strings)
        with torch.no_grad():
            o = txth(torch.from_numpy(emb).to(enc.device))
        return pack_bits(o[bi]["binary"].cpu().numpy())

    # hold-out test queries
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en_caps = [str(c) for c in EC["test"]["captions"]]
    te_en = EC["test"]["txt"].float().numpy()
    te_img = EC["test"]["img"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    ko_full = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); ko_full[_cocoid(e["image_path"])] = e.get("captions", [])
    ko_caps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]

    gal = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))   # frozen image codes
    gal_ix = _faiss(gal)
    packed = np.fromfile(Path(args.static) / "data/index.bin", dtype=np.uint8).reshape(-1, CODE_BYTES)
    idx50 = _faiss(np.ascontiguousarray(packed))
    rng = np.random.default_rng(42)
    osel = np.sort(rng.choice(len(te_ids), min(args.n_overlap, len(te_ids)), replace=False))

    srv = {"EN": pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(te_en))),
           "KO": pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(ko_emb)))}
    srv_R = {L: _recall(gal_ix, srv[L], te_ids, te_ids) for L in ("EN", "KO")}
    print(f"[c1-eval] SERVER so400m R@10: EN {srv_R['EN']} | KO {srv_R['KO']} (= 79.92/71.08)", flush=True)

    caps = {"EN": en_caps, "KO": ko_caps}
    results = {"server_R10": srv_R, "cycle4_ref": C4}
    for L in ("EN", "KO"):
        t0 = time.perf_counter()
        oc = offline_codes(caps[L])
        ms = (time.perf_counter() - t0) / len(caps[L]) * 1e3
        r10 = _recall(gal_ix, oc, te_ids, te_ids)
        _, Io = idx50.search(oc[osel], 10)
        _, Iv = idx50.search(srv[L][osel], 10)
        ov = float(np.mean([len(set(Io[i]) & set(Iv[i])) / 10 for i in range(len(osel))]))
        ham_off = _pair_hamming(oc, gal)            # offline text vs its own image (5K aligned)
        ham_srv = _pair_hamming(srv[L], gal)        # server text vs its own image
        results[L] = {
            "R10": r10, "R10_delta_vs_server": round(r10 - srv_R[L], 2),
            "R10_delta_vs_cycle4": round(r10 - C4[L]["R10"], 2),
            "top10_overlap": round(ov, 3), "overlap_delta_vs_cycle4": round(ov - C4[L]["overlap"], 3),
            "pair_hamming_offline": round(ham_off, 1), "pair_hamming_server": round(ham_srv, 1),
            "ms_per_query": round(ms, 1)}
        print(f"[c1-eval] {L}: R@10 {r10} (Δsrv {results[L]['R10_delta_vs_server']:+}, "
              f"Δc4 {results[L]['R10_delta_vs_cycle4']:+}) | overlap {round(ov,3)} "
              f"(Δc4 {results[L]['overlap_delta_vs_cycle4']:+}) | pairHam off {ham_off:.1f}/srv {ham_srv:.1f} "
              f"/1024 | {results[L]['ms_per_query']}ms/q", flush=True)

    print("[c1-eval] RESULT_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
