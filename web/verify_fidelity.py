"""Fidelity verification (cycle 2) — prove the demo's LIVE query encoding reproduces the
cached embeddings that scripts/eval_korean.py used for KO R@10 71 / EN 80.

This is the missing link: v1's parity tests only proved internal consistency
(JS == python-mirror == faiss over index.bin, server == offline Encoder). None checked
that `web/common.py:Encoder` reproduces the *eval* embeddings. Here we do, with the DEMO
encoder exactly as the server runs it (bf16 text tower on GPU):

  EN : live-encode emb_cache.pt['test']['captions']  vs cached EC['test']['txt']
  KO : live-encode coco_ko.jsonl test caps (rebuilt like build_korean.py, aligned to ids)
       vs coco_ko_test.pt['txt_emb']
       -> per-string cosine (primary; >= --thresh) AND packed-code agreement through txt_h
  IMG: image_codes_packed(te_img) vs eval_korean's img_h path -> expect BYTE-IDENTICAL

NOTE the known asymmetry: eval caches were built with an fp32 text tower (build_korean.py),
the demo runs a bf16 tower (common.py on cuda). The head (txt_h/img_h) is fp32 on both, so
the image codes must match exactly; text cosine quantifies the bf16-vs-fp32 gap.

VERIFICATION ONLY — does not modify web/common.py. Doubles as the regression guard:
fixed seed/N/threshold, prints PASS/FAIL + numbers, non-zero exit on failure.

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/verify_fidelity.py [--n 1000] [--thresh 0.999]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BYTES, Encoder, pack_bits  # noqa: E402

_ID_RE = re.compile(r"_0*(\d+)\.jpg")


def _cocoid(path: str) -> int:
    m = _ID_RE.search(path)
    return int(m.group(1)) if m else -1


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between (N,D) arrays."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(1)


def _stats(name: str, cos: np.ndarray, thresh: float) -> dict:
    frac = float((cos >= thresh).mean())
    d = {"n": int(cos.size), "min": float(cos.min()), "mean": float(cos.mean()),
         "median": float(np.median(cos)), "p1": float(np.percentile(cos, 1)),
         "frac_ge_thresh": frac}
    print(f"  [{name}] n={d['n']} cos: min={d['min']:.5f} p1={d['p1']:.5f} "
          f"median={d['median']:.5f} mean={d['mean']:.5f} | frac>={thresh}: {frac*100:.1f}%",
          flush=True)
    return d


def _code_agreement(name: str, demo_codes: np.ndarray, eval_codes: np.ndarray, bits: int) -> dict:
    """Per-row packed-code agreement: % byte-identical + mean Hamming (bits)."""
    identical = int((demo_codes == eval_codes).all(axis=1).sum())
    xor = np.bitwise_xor(demo_codes, eval_codes)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    ham = lut[xor].sum(1)
    d = {"n": int(len(demo_codes)), "byte_identical": identical,
         "pct_identical": round(100 * identical / len(demo_codes), 2),
         "mean_hamming": round(float(ham.mean()), 3), "max_hamming": int(ham.max()), "bits": bits}
    print(f"  [{name}] packed codes: {identical}/{len(demo_codes)} byte-identical "
          f"({d['pct_identical']}%) | mean Hamming {d['mean_hamming']}/{bits} "
          f"(max {d['max_hamming']})", flush=True)
    return d


def main() -> None:
    p = argparse.ArgumentParser(description="Fidelity: demo encoding vs eval_korean caches")
    p.add_argument("--n", type=int, default=1000, help="per-language sample (0 = all 5000)")
    p.add_argument("--thresh", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb-cache", default="/tmp/emb_cache.pt")
    p.add_argument("--ko-cache", default="/tmp/coco_ko_test.pt")
    p.add_argument("--ko-jsonl", default="data/coco_ko/coco_ko.jsonl")
    args = p.parse_args()

    import torch

    EC = torch.load(args.emb_cache, map_location="cpu")
    KO = torch.load(args.ko_cache, map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    ko_ids = [int(x) for x in KO["ids"].tolist()]
    if ko_ids != te_ids:
        raise SystemExit("KO ids do not align to emb_cache test ids — caches mismatched")
    en_caps = [str(c) for c in EC["test"]["captions"]]
    te_en = EC["test"]["txt"].float().numpy()
    te_img = EC["test"]["img"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    n_total = len(te_ids)

    # Reconstruct KO test strings exactly like build_korean.py (first KO cap per cocoid).
    ko_full: dict[int, list] = {}
    kp = Path(args.ko_jsonl)
    if not kp.is_absolute():
        kp = Path(REPO) / args.ko_jsonl
    if kp.exists():
        for line in open(kp):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ko_full[_cocoid(e["image_path"])] = e.get("captions", [])
    ko_caps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]
    ko_missing = sum(1 for c in ko_caps if not c)
    print(f"[fidelity] test set: {n_total} | EN caps {len(en_caps)} | "
          f"KO caps rebuilt from {kp} (missing={ko_missing})", flush=True)

    rng = np.random.default_rng(args.seed)
    sample = (np.arange(n_total) if args.n <= 0 or args.n >= n_total
              else np.sort(rng.choice(n_total, args.n, replace=False)))
    print(f"[fidelity] sampling {len(sample)} of {n_total} per language (seed {args.seed})\n",
          flush=True)

    enc = Encoder()
    print(f"[fidelity] demo Encoder: head={os.path.basename(enc.head_path)} "
          f"norm_in={enc.norm_in} device={enc.device} | comparing to fp32 eval caches\n",
          flush=True)

    results = {"thresh": args.thresh, "n_sample": int(len(sample))}
    overall_ok = True

    for name, caps, cached in [("EN", en_caps, te_en), ("KO", ko_caps, ko_emb)]:
        live = np.zeros((len(sample), te_en.shape[1]), dtype=np.float32)
        demo_codes = np.zeros((len(sample), CODE_BYTES), dtype=np.uint8)
        eval_codes = np.zeros((len(sample), CODE_BYTES), dtype=np.uint8)
        for j, i in enumerate(sample):
            le = enc.encode_text_emb(caps[int(i)])                   # demo live tower (1,1152), bf16
            live[j] = le[0]
            # demo packed code from the SAME live emb (== text_code_packed, no 2nd tower call)
            demo_codes[j] = pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(le)))[0]
            eval_codes[j] = pack_bits(enc._codes_pm1(                 # cached emb through SAME txt_h
                enc.txt_h, torch.from_numpy(cached[int(i):int(i) + 1])))[0]
        print(f"[fidelity] {name}: live-encode vs cached embedding", flush=True)
        cstat = _stats(name, _cos_rows(live, cached[sample]), args.thresh)
        astat = _code_agreement(name, demo_codes, eval_codes, enc.bits)
        ok = cstat["mean"] >= args.thresh and cstat["frac_ge_thresh"] >= 0.99
        results[name] = {"cosine": cstat, "code": astat, "pass": ok}
        overall_ok = overall_ok and ok
        print(f"  [{name}] -> {'PASS' if ok else 'FAIL'} "
              f"(mean cos >= {args.thresh} and >=99% of samples)\n", flush=True)

    # Image side: demo image codes vs eval_korean's img_h path — expect byte-identical.
    print("[fidelity] IMAGE: demo image_codes_packed vs eval_korean img_h path", flush=True)
    demo_img = enc.image_codes_packed(te_img)
    eval_img = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))
    img_ident = bool((demo_img == eval_img).all())
    img_mismatch = int((~(demo_img == eval_img).all(axis=1)).sum())
    print(f"  [IMG] {n_total} rows: byte-identical={img_ident} (mismatched rows={img_mismatch})",
          flush=True)
    results["image"] = {"byte_identical": img_ident, "mismatched_rows": img_mismatch}
    overall_ok = overall_ok and img_ident

    print(f"\n[fidelity] {'PASS' if overall_ok else 'FAIL'} "
          f"(EN+KO cosine fidelity & image byte-identity)", flush=True)
    print("[fidelity] RESULT_JSON " + json.dumps(results), flush=True)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
