"""Parity verification (Acceptance Criteria §6.4 / §6.5).

Proves the two claims the whole demo rests on:

  §6.4  The client-side Hamming search reproduces faiss `IndexBinaryFlat` on the SAME
        index.bin. For >=20 KO/EN queries we compare, per query:
          (a) faiss IndexBinaryFlat top-10, and
          (b) a pure-Python re-implementation of app.js's search (256-entry uint8
              popcount LUT; dist = sum(LUT[a^b]); ties broken by ascending index).
        Comparison is tie-immune: both result lists are re-sorted by (distance, id)
        before comparing, so a genuine packing/metric mismatch is the only way to fail.

  §6.5  /encode_query returns the same code as the offline pipeline. With --server URL,
        each query is also encoded by the running server and the returned bytes are
        compared to the local Encoder (the path build_index.py used).

Run on DGX (loads the real artifacts the browser loads):
  cd ~/github/vlm_quantization
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/verify_parity.py \
      --static web/static [--server http://127.0.0.1:8300] [--k 10]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BITS, CODE_BYTES, Encoder, hamming_topk

# >=20 queries spanning Korean + English, mixed concepts.
QUERIES = [
    "바닷가 강아지", "눈 덮인 산", "도시의 야경", "빨간 우산을 든 사람", "기차역 플랫폼",
    "해변에서 노는 아이들", "주방에서 요리하는 사람", "노란 꽃밭", "고양이 두 마리",
    "비 오는 거리", "축구하는 사람들", "커피 한 잔", "다리 위의 자동차", "숲 속의 오두막",
    "a dog on the beach", "a snowy mountain", "city skyline at night", "a plate of food",
    "people riding bicycles", "a baby elephant", "an old wooden boat", "a slice of pizza",
    "two giraffes", "a person surfing a wave",
]


def load_index(static: Path):
    data = static / "data"
    packed = np.fromfile(data / "index.bin", dtype=np.uint8)
    if packed.size % CODE_BYTES != 0:
        raise SystemExit(f"index.bin size {packed.size} not a multiple of {CODE_BYTES}")
    packed = np.ascontiguousarray(packed.reshape(-1, CODE_BYTES))
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    if len(meta) != packed.shape[0]:
        raise SystemExit(f"ALIGNMENT BROKEN: meta {len(meta)} != index rows {packed.shape[0]}")
    return packed, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Verify client search == faiss")
    p.add_argument("--static", default="web/static")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--server", default=None, help="optional running query_server URL")
    p.add_argument("--dump-js-queries", default=None,
                   help="write [{text, code_b64, faiss_pairs}] JSON for the Node JS parity test")
    args = p.parse_args()

    packed, meta = load_index(Path(args.static))
    n = packed.shape[0]
    print(f"[verify] index.bin: {n:,} rows x {CODE_BYTES} B  (meta aligned: OK)", flush=True)

    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    fidx = faiss.IndexBinaryFlat(CODE_BITS)
    fidx.add(packed)

    enc = Encoder()
    server_ok = server_total = 0
    if args.server:
        import urllib.request

    matched = 0
    mirror_ms = []
    mismatches = []
    dump = []
    for q in QUERIES:
        code = enc.text_code_packed(q)  # (CODE_BYTES,) uint8 — offline/server code path

        # (a) faiss top-k
        fd, fi = fidx.search(code[None, :], args.k)
        faiss_pairs = sorted(zip(fd[0].tolist(), fi[0].tolist()))  # (dist, id) asc
        dump.append({"text": q, "code_b64": base64.b64encode(code.tobytes()).decode("ascii"),
                     "faiss_pairs": faiss_pairs})

        # (b) JS-mirror top-k (timed)
        t0 = time.perf_counter()
        m_idx, m_dist = hamming_topk(packed, code, args.k)
        mirror_ms.append((time.perf_counter() - t0) * 1e3)
        mirror_pairs = sorted(zip(m_dist.tolist(), m_idx.tolist()))

        if faiss_pairs == mirror_pairs:
            matched += 1
        else:
            mismatches.append((q, faiss_pairs, mirror_pairs))

        # §6.5: server-returned code must equal the local (offline) code byte-for-byte
        if args.server:
            server_total += 1
            try:
                req = urllib.request.Request(
                    args.server.rstrip("/") + "/encode_query",
                    data=json.dumps({"text": q}).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
                srv = np.frombuffer(base64.b64decode(resp["code"]), dtype=np.uint8)
                if srv.shape == code.shape and bool((srv == code).all()):
                    server_ok += 1
            except Exception as e:  # pragma: no cover
                print(f"[verify] server check failed for {q!r}: {e}", flush=True)

    print(f"\n[verify] §6.4 client(JS-mirror) vs faiss IndexBinaryFlat top-{args.k}: "
          f"{matched}/{len(QUERIES)} exact match", flush=True)
    print(f"[verify] §6.3 JS-mirror search time over {n:,} rows: "
          f"mean {np.mean(mirror_ms):.1f} ms, max {np.max(mirror_ms):.1f} ms "
          f"(numpy proxy; browser logs its own time)", flush=True)
    if args.server:
        print(f"[verify] §6.5 /encode_query == offline code: {server_ok}/{server_total} "
              f"byte-exact", flush=True)
    for q, fp, mp in mismatches[:5]:
        print(f"  MISMATCH {q!r}\n    faiss : {fp}\n    mirror: {mp}", flush=True)

    if args.dump_js_queries:
        with open(args.dump_js_queries, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False)
        print(f"[verify] dumped {len(dump)} queries -> {args.dump_js_queries} "
              f"(for web/verify_js.mjs)", flush=True)

    ok = (matched == len(QUERIES)) and (not args.server or server_ok == server_total)
    print("\n[verify] PARITY OK" if ok else "\n[verify] PARITY FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
