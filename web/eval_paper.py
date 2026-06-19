"""Paper supplementary measurements — EVAL ONLY (no retraining, no index rebuild, no head
or common.py changes). Reuses frozen ft113 head + cached so400m embeddings on the
eval_korean 5K protocol: gallery = img_h(te_img) (5K test images), gold = caption_i -> image_i.

Produces:
  paper/upper_bound.csv  (A: float/cosine ceiling)
  paper/bits_sweep.csv   (B: nested-code bit-length sweep — server + offline C1)
  paper/precision.csv    (C: precision sensitivity — head / emb / text-tower bf16)

Anchor: 1024-bit server EN R@10 79.92 / KO 71.08 must reproduce (printed up top).

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/eval_paper.py [--txt-head /tmp/txt_h_e5.pt]
"""
from __future__ import annotations

import argparse
import copy
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

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
_ID = re.compile(r"_0*(\d+)\.jpg")
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
KS = (1, 5, 10)


def nrm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def recall_ks(ix, q, gold_rows, ks=KS):
    _, I = ix.search(q, max(ks))
    return {k: round(100 * np.mean([gold_rows[i] in I[i, :k] for i in range(len(gold_rows))]), 2) for k in ks}


def pair_bitflip(a, b):
    return float(_LUT[np.bitwise_xor(a, b)].sum(1).mean())


def head_codes(head, emb_np, enc, bit, device=None, in_dtype=None):
    """head(prep(emb)) -> ±1 numpy at the given bit length."""
    dev = device or enc.device
    et = enc._prep(torch.from_numpy(np.ascontiguousarray(emb_np, dtype=np.float32)).to(dev))
    if in_dtype is not None:
        et = et.to(in_dtype)
    with torch.no_grad():
        outs = head(et)
    bi = [int(b) for b in enc.bit_list].index(bit)
    return outs[bi]["binary"].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt-head", default="/tmp/txt_h_e5.pt")
    args = ap.parse_args()
    PAPER.mkdir(parents=True, exist_ok=True)

    enc = Encoder()
    bit_list = [int(b) for b in enc.bit_list]
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy()
    te_en = EC["test"]["txt"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    n = len(te_ids)
    gold = list(range(n))  # gallery row i is the gold image for query i
    TXT = {"EN": te_en, "KO": ko_emb}
    print(f"[paper] test {n} | bits {bit_list} | device {enc.device}", flush=True)

    # ---- anchor: 1024-bit server ----
    gal1024 = pack_bits(head_codes(enc.img_h, te_img, enc, CODE_BITS))
    ix1024 = faiss_bin(gal1024, CODE_BITS)
    anchor = {L: recall_ks(ix1024, pack_bits(head_codes(enc.txt_h, TXT[L], enc, CODE_BITS)), gold)
              for L in ("EN", "KO")}
    print(f"[paper] ANCHOR 1024 server R@10: EN {anchor['EN'][10]} / KO {anchor['KO'][10]} "
          f"(expect 79.92 / 71.08)", flush=True)

    # ===== A. float ceiling (cosine) =====
    img_n = nrm(te_img)
    A = {}
    for L in ("EN", "KO"):
        sims = nrm(TXT[L]) @ img_n.T
        A[L] = {k: round(100 * np.mean([gold[i] in np.argpartition(-sims[i], k)[:k] for i in range(n)]), 2) for k in KS}
    with open(PAPER / "upper_bound.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "EN_R1", "EN_R5", "EN_R10", "KO_R1", "KO_R5", "KO_R10"])
        w.writerow(["float_cosine_ceiling", A["EN"][1], A["EN"][5], A["EN"][10], A["KO"][1], A["KO"][5], A["KO"][10]])
        w.writerow(["bit1024_server", anchor["EN"][1], anchor["EN"][5], anchor["EN"][10],
                    anchor["KO"][1], anchor["KO"][5], anchor["KO"][10]])
    print(f"[paper] A float ceiling: EN R@10 {A['EN'][10]} / KO {A['KO'][10]} "
          f"(>= 1bit server {anchor['EN'][10]}/{anchor['KO'][10]}: "
          f"{A['EN'][10] >= anchor['EN'][10] and A['KO'][10] >= anchor['KO'][10]})", flush=True)

    # ===== B. bit-length sweep (server + offline C1) =====
    SWEEP = [b for b in (64, 128, 256, 512, 1024) if b in bit_list]
    rows = []
    # offline txt head (C1) if available
    offl = None
    if os.path.exists(args.txt_head):
        from web.headadapt_eval import E5
        ck = torch.load(args.txt_head, map_location="cpu")
        th = NestedHashLayer(ck["embed"], ck["hidden"], [int(b) for b in ck["bits"]], 0.0)
        th.load_state_dict(ck["txt_h"]); th.to(enc.device).eval()
        e5 = E5(ck["student"], enc.device)
        ko_full = {}
        for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
            e = json.loads(line)
            m = _ID.search(e["image_path"]); ko_full[int(m.group(1)) if m else -1] = e.get("captions", [])
        en_caps = [str(c) for c in EC["test"]["captions"]]
        ko_caps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]
        e5emb = {"EN": e5.embed(en_caps), "KO": e5.embed(ko_caps)}
        offl = (th, e5emb)
    for bit in SWEEP:
        gal = pack_bits(head_codes(enc.img_h, te_img, enc, bit))
        ix = faiss_bin(gal, bit)
        row = {"bits": bit, "bytes_per_img": bit // 8, "index_MB_50k": round(50000 * (bit // 8) / 1e6, 2)}
        for L in ("EN", "KO"):
            R = recall_ks(ix, pack_bits(head_codes(enc.txt_h, TXT[L], enc, bit)), gold)
            row[f"{L}_R1"], row[f"{L}_R5"], row[f"{L}_R10"] = R[1], R[5], R[10]
        if offl:
            th, e5emb = offl
            for L in ("EN", "KO"):
                R = recall_ks(ix, pack_bits(head_codes(th, e5emb[L], enc, bit)), gold)
                row[f"off_{L}_R1"], row[f"off_{L}_R5"], row[f"off_{L}_R10"] = R[1], R[5], R[10]
        rows.append(row)
        print(f"[paper] B bit {bit}: server EN/KO R@10 {row['EN_R10']}/{row['KO_R10']}"
              + (f" | offline {row.get('off_EN_R10')}/{row.get('off_KO_R10')}" if offl else ""), flush=True)
    with open(PAPER / "bits_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ===== C. precision sensitivity (1024-bit server) =====
    fp32_img = head_codes(enc.img_h, te_img, enc, CODE_BITS)        # ±1 ref
    fp32_txt = {L: head_codes(enc.txt_h, TXT[L], enc, CODE_BITS) for L in ("EN", "KO")}
    ref_img_p = pack_bits(fp32_img)
    ref_txt_p = {L: pack_bits(fp32_txt[L]) for L in ("EN", "KO")}
    ix_ref = faiss_bin(ref_img_p, CODE_BITS)

    def top10_overlap(qa, qb, ix):
        _, Ia = ix.search(qa, 10); _, Ib = ix.search(qb, 10)
        return round(float(np.mean([len(set(Ia[i]) & set(Ib[i])) / 10 for i in range(len(qa))])), 3)

    prec = []

    def emit(target, dtype, img_codes, txt_codes, gal_p, note=""):
        ix = faiss_bin(gal_p, CODE_BITS)
        row = {"target": target, "dtype": dtype, "note": note}
        bf, ov, r10 = [], [], {}
        for L in ("EN", "KO"):
            tp = pack_bits(txt_codes[L])
            bf.append(pair_bitflip(tp, ref_txt_p[L]))
            ov.append(top10_overlap(tp, ref_txt_p[L], ix_ref))
            r10[L] = recall_ks(ix, tp, gold)[10]
        row["bitflips_per_1024"] = round(float(np.mean(bf)), 2)
        row["top10_overlap"] = round(float(np.mean(ov)), 3)
        row["EN_R10"], row["KO_R10"] = r10["EN"], r10["KO"]
        prec.append(row)
        print(f"[paper] C {target} {dtype}: flip {row['bitflips_per_1024']}/1024 "
              f"overlap {row['top10_overlap']} R@10 {r10['EN']}/{r10['KO']} {note}", flush=True)

    # baseline
    emit("head", "fp32", fp32_img, fp32_txt, ref_img_p, "baseline")
    # C-head: cast both heads to dtype
    for dt in ("fp16", "bf16", "int8"):
        try:
            if dt == "int8":
                import torch.ao.quantization as q
                ih = q.quantize_dynamic(copy.deepcopy(enc.img_h).cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
                th = q.quantize_dynamic(copy.deepcopy(enc.txt_h).cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
                ic = head_codes(ih, te_img, enc, CODE_BITS, device="cpu")
                tc = {L: head_codes(th, TXT[L], enc, CODE_BITS, device="cpu") for L in ("EN", "KO")}
            else:
                d = torch.float16 if dt == "fp16" else torch.bfloat16
                ih = copy.deepcopy(enc.img_h).to(enc.device, d).eval()
                th = copy.deepcopy(enc.txt_h).to(enc.device, d).eval()
                ic = head_codes(ih, te_img, enc, CODE_BITS, in_dtype=d)
                tc = {L: head_codes(th, TXT[L], enc, CODE_BITS, in_dtype=d) for L in ("EN", "KO")}
            emit("head", dt, ic, tc, pack_bits(ic))
        except Exception as e:
            prec.append({"target": "head", "dtype": dt, "note": f"FAILED {repr(e)[:80]}"})
            print(f"[paper] C head {dt}: FAILED {repr(e)[:80]}", flush=True)
    # C-emb: cast cached embeddings to dtype (encoder-output storage precision), fp32 head
    def cast_emb(x, dt):
        if dt == "int8":
            s = np.abs(x).max(axis=1, keepdims=True) / 127.0 + 1e-9
            return (np.round(x / s).clip(-127, 127) * s).astype(np.float32)
        d = np.float16 if dt == "fp16" else None
        if d is not None:
            return x.astype(np.float16).astype(np.float32)
        return torch.from_numpy(x).to(torch.bfloat16).float().numpy()  # bf16
    for dt in ("fp16", "bf16", "int8"):
        try:
            ic = head_codes(enc.img_h, cast_emb(te_img, dt), enc, CODE_BITS)
            tc = {L: head_codes(enc.txt_h, cast_emb(TXT[L], dt), enc, CODE_BITS) for L in ("EN", "KO")}
            emit("emb", dt, ic, tc, pack_bits(ic), "storage-cast")
        except Exception as e:
            prec.append({"target": "emb", "dtype": dt, "note": f"FAILED {repr(e)[:80]}"})
            print(f"[paper] C emb {dt}: FAILED", flush=True)

    # C-text-tower bf16 (faithful to cycle 2): so400m TOWER computed in bf16 vs the fp32 cache,
    # fp32 head, image gallery fp32. This is the "~2.7/1024 flip" condition from the outline.
    try:
        en_caps = [str(c) for c in EC["test"]["captions"]]
        kf = {}
        for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
            e = json.loads(line); m = _ID.search(e["image_path"])
            kf[int(m.group(1)) if m else -1] = e.get("captions", [])
        ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
        rng = np.random.default_rng(42); samp = np.sort(rng.choice(n, 1000, replace=False))
        bf, r10s = [], {}
        for L, caps in [("EN", en_caps), ("KO", ko_caps)]:
            live = np.concatenate([enc.encode_text_emb(caps[int(i)]) for i in samp], 0)  # bf16 cuda tower
            live_p = pack_bits(head_codes(enc.txt_h, live, enc, CODE_BITS))
            bf.append(pair_bitflip(live_p, ref_txt_p[L][samp]))
            r10s[L] = recall_ks(ix1024, live_p, [int(i) for i in samp])[10]
        prec.append({"target": "text_tower", "dtype": "bf16",
                     "bitflips_per_1024": round(float(np.mean(bf)), 2), "top10_overlap": "",
                     "EN_R10": r10s["EN"], "KO_R10": r10s["KO"],
                     "note": "so400m tower bf16(cuda) vs fp32 cache, head fp32, n=1000/lang sample"})
        print(f"[paper] C text_tower bf16: flip {round(float(np.mean(bf)), 2)}/1024 "
              f"R@10(n=1000) {r10s['EN']}/{r10s['KO']}", flush=True)
    except Exception as e:
        print(f"[paper] C text_tower bf16: FAILED {repr(e)[:80]}", flush=True)

    with open(PAPER / "precision.csv", "w", newline="") as f:
        cols = ["target", "dtype", "bitflips_per_1024", "top10_overlap", "EN_R10", "KO_R10", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(prec)

    print("[paper] RESULT_JSON " + json.dumps({"anchor": {"EN": anchor["EN"][10], "KO": anchor["KO"][10]},
          "float": {"EN": A["EN"][10], "KO": A["KO"][10]}, "precision_rows": len(prec)}), flush=True)
    print("[paper] A/B/C DONE -> paper/{upper_bound,bits_sweep,precision}.csv", flush=True)


if __name__ == "__main__":
    main()
