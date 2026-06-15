"""Side-by-side demo server: float cosine vs binary-hash (Hamming) image search.

Design for honest latency comparison:
  POST /encode {q}            -> encodes the text query ONCE (SigLIP2 text head +
                                 hash head), caches it, returns {qid, encode_ms}.
  GET  /search/float?qid=&k=  -> float cosine search only         -> {results, search_ms}
  GET  /search/bit?qid=&bit=&k= -> binary Hamming search only     -> {results, search_ms}

The frontend calls /encode once, then fires /search/float and /search/bit in
PARALLEL and renders whichever returns first — so the (much faster) Hamming search
visibly lands before the float scan. The shared encode cost is shown separately.

A tiled corpus (CORPUS_MULT) inflates the index so the search step is non-trivial
and the speed gap is felt. Result positions are mapped back to real images
(de-duplicated), and the true corpus size is reported honestly.

Env: DEMO_INDEX, DEMO_HEADS, DEMO_IMAGE_ROOT, CORPUS_MULT, REPO.
Run:  uvicorn demo.server:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations
import os, sys, time, uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization")
sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

MODEL = "google/siglip2-so400m-patch14-384"
INDEX = os.environ.get("DEMO_INDEX", "/tmp/demo_index.npz")
HEADS = os.environ.get("DEMO_HEADS", "/tmp/demo_hashheads.pt")
IMAGE_ROOT = os.environ.get("DEMO_IMAGE_ROOT", f"{REPO}/data/coco")
MULT = int(os.environ.get("CORPUS_MULT", "200"))

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32

print(f"[demo] loading hash heads + SigLIP2 text model on {dev} ...", flush=True)
hh = torch.load(HEADS, map_location="cpu")
BITS = hh["bits"]
txt_h = NestedHashLayer(hh["embed"], hh["hidden"], BITS, 0.1)
txt_h.load_state_dict(hh["txt_h"]); txt_h.to(dev).eval()

from transformers import AutoModel
backbone = AutoModel.from_pretrained(MODEL, dtype=dt).to(dev).eval()
try:
    del backbone.vision_model  # text-only at query time
    torch.cuda.empty_cache()
except Exception:
    pass
try:
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer
    tok = GemmaTokenizer.from_pretrained(MODEL)

idx = np.load(INDEX, allow_pickle=True)
n_real = int(len(idx["ids"]))
emb = idx["emb"].astype(np.float32)               # (n,1152) normalized
packed = {b: idx[f"packed_{b}"] for b in BITS}     # (n, b/8) uint8
paths = idx["paths"]; captions = idx["captions"]
emb_t = np.ascontiguousarray(np.tile(emb, (MULT, 1)))
packed_t = {b: np.ascontiguousarray(np.tile(packed[b], (MULT, 1))) for b in BITS}
n_corpus = int(emb_t.shape[0])
print(f"[demo] corpus: {n_real} real x{MULT} = {n_corpus} | bits {BITS}", flush=True)

# faiss IndexBinaryFlat: hardware POPCNT + multithreaded, keeps packed 32B/item.
# Measured on GB10: ~5x (256-bit) to ~6x (1024-bit) faster than numpy at 1M corpus,
# while preserving the storage advantage (no unpacking). numpy is the fallback.
try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    faiss = None
    _HAVE_FAISS = False

bin_index: dict[int, "faiss.IndexBinaryFlat"] = {}
if _HAVE_FAISS:
    for b in BITS:
        ix = faiss.IndexBinaryFlat(b)
        ix.add(packed_t[b])                 # stores a copy of the packed codes (b/8 B/item)
        bin_index[b] = ix
    print(f"[demo] faiss IndexBinaryFlat ready (threads={faiss.omp_get_max_threads()})", flush=True)
else:
    print("[demo] faiss unavailable -> numpy uint64 popcount fallback", flush=True)

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
_HAS_BITCOUNT = hasattr(np, "bitwise_count")  # numpy >= 2.0: C-level popcount (≈5x LUT)


def _popcount_sum(db: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamming distance of each row of `db` to `q` (both packed uint8).

    Views the packed bytes as uint64 when the width is a multiple of 8 bytes:
    the reduction then runs over 8x fewer elements (measured ~1.5-2.5x on GB10).
    """
    if db.shape[1] % 8 == 0:
        db = db.view(np.uint64); q = q.view(np.uint64)
    xor = np.bitwise_xor(db, q)
    if _HAS_BITCOUNT:
        return np.bitwise_count(xor).sum(axis=1)
    return _POP[xor.view(np.uint8)].sum(axis=1)


CACHE: dict[str, dict] = {}


def _pool(o):
    return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)


def _encode(q: str):
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    ii = t["input_ids"].to(dev); am = t.get("attention_mask")
    am = am.to(dev) if am is not None else None
    with torch.no_grad():
        e = _pool(backbone.text_model(input_ids=ii, attention_mask=am)).float()
        outs = txt_h(e.to(dev))
    e_norm = F.normalize(e, dim=1)[0].cpu().numpy().astype(np.float32)
    codes = {}
    for k, b in enumerate(BITS):
        c01 = (outs[k]["binary"][0] > 0).cpu().numpy().astype(np.uint8)
        codes[b] = np.packbits(c01)
    return e_norm, codes


def _unique_topk(order: np.ndarray, k: int) -> list[int]:
    seen, out = set(), []
    for p in order:
        r = int(p) % n_real
        if r not in seen:
            seen.add(r); out.append(r)
            if len(out) >= k:
                break
    return out


def _results(rows: list[int]) -> list[dict]:
    return [{"position": r, "path": str(paths[r]), "caption": str(captions[r])} for r in rows]


app = FastAPI(title="Float vs Binary-Hash Image Search Demo")

from fastapi.staticfiles import StaticFiles
if Path(IMAGE_ROOT).is_dir():
    app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

_DEMO_HTML = Path(__file__).resolve().parent / "compare.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return _DEMO_HTML.read_text(encoding="utf-8") if _DEMO_HTML.exists() else "<h1>compare.html missing</h1>"


@app.get("/stats")
def stats():
    return {"real": n_real, "mult": MULT, "corpus": n_corpus, "bits": BITS,
            "float_bytes_per_item": emb.shape[1] * 4,
            "bit_bytes_per_item": {b: b // 8 for b in BITS}}


class EncReq(BaseModel):
    q: str


@app.post("/encode")
def encode(r: EncReq):
    t0 = time.perf_counter()
    e_norm, codes = _encode(r.q)
    ms = (time.perf_counter() - t0) * 1e3
    qid = uuid.uuid4().hex[:12]
    CACHE[qid] = {"emb": e_norm, "codes": codes, "q": r.q}
    if len(CACHE) > 256:
        CACHE.pop(next(iter(CACHE)))
    return {"qid": qid, "encode_ms": round(ms, 2), "q": r.q}


@app.get("/search/float")
def search_float(qid: str, k: int = 12):
    c = CACHE.get(qid)
    if not c:
        raise HTTPException(404, "qid expired; re-encode")
    q = c["emb"]
    t0 = time.perf_counter()
    scores = emb_t @ q                      # cosine (emb already normalized)
    take = min(k * MULT + k, n_corpus - 1)
    order = np.argpartition(-scores, take - 1)[:take]   # single kth -> O(N)
    order = order[np.argsort(-scores[order])]
    rows = _unique_topk(order, k)
    ms = (time.perf_counter() - t0) * 1e3
    return {"mode": "float", "search_ms": round(ms, 3), "corpus": n_corpus,
            "bytes_per_item": emb.shape[1] * 4, "results": _results(rows)}


@app.get("/search/bit")
def search_bit(qid: str, bit: int = 256, k: int = 12):
    c = CACHE.get(qid)
    if not c:
        raise HTTPException(404, "qid expired; re-encode")
    if bit not in BITS:
        raise HTTPException(400, f"bit {bit} not in {BITS}")
    qcode = c["codes"][bit]
    take = min(k * MULT + k, n_corpus)
    if _HAVE_FAISS:
        backend = "faiss IndexBinaryFlat (HW POPCNT, multithread)"
        t0 = time.perf_counter()
        _, ids = bin_index[bit].search(qcode.reshape(1, -1), take)  # sorted by Hamming
        order = ids[0]
        rows = _unique_topk(order, k)
        ms = (time.perf_counter() - t0) * 1e3
    else:
        backend = "numpy uint64 popcount"
        t0 = time.perf_counter()
        dist = _popcount_sum(packed_t[bit], qcode)      # Hamming distance
        order = np.argpartition(dist, take - 1)[:take]  # single kth -> O(N)
        order = order[np.argsort(dist[order])]
        rows = _unique_topk(order, k)
        ms = (time.perf_counter() - t0) * 1e3
    return {"mode": f"{bit}-bit", "search_ms": round(ms, 3), "corpus": n_corpus,
            "bytes_per_item": bit // 8, "backend": backend, "results": _results(rows)}
