"""Standalone cross-modal image-retrieval API (separate from the monitor dashboard).

Loads a trained checkpoint (query encoder) plus a packed serving index (corpus of
image hash codes) and answers nearest-neighbour queries by Hamming distance.

    text query  -> hash code -> Hamming search over image index   (text-to-image)
    image query -> hash code -> Hamming search over image index   (image-to-image)

Run:
    export RETRIEVAL_CHECKPOINT=checkpoints/<run>/best.ckpt
    export RETRIEVAL_INDEX=indexes/serving.npz
    uvicorn src.serve.api:app --host 0.0.0.0 --port 8100

Endpoints:
    GET  /healthz          liveness + load state
    GET  /stats            index size / available bit lengths
    POST /search/text      {"query": "...", "k": 10, "bit": 64}
    POST /search/image     {"image_b64": "...", "k": 10, "bit": 64}

Both search responses include `took_ms` so callers can see the search latency.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.serve.binary_index import ServingIndex

app = FastAPI(title="VLM Hash Retrieval", version="1.0")

# Optional demo UI: serve the photo-search frontend at "/" and the corpus images
# at "/images" when RETRIEVAL_IMAGE_ROOT is set (so result paths render in-browser).
_IMAGE_ROOT = os.environ.get("RETRIEVAL_IMAGE_ROOT")
if _IMAGE_ROOT:
    from fastapi.staticfiles import StaticFiles

    app.mount("/images", StaticFiles(directory=_IMAGE_ROOT), name="images")

_DEMO_HTML = Path(__file__).resolve().parents[2] / "demo" / "index.html"


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    """Self-contained photo-search demo UI (vanilla JS -> /stats + /search/text)."""
    if _DEMO_HTML.exists():
        return _DEMO_HTML.read_text(encoding="utf-8")
    return "<h1>VLM Hash Retrieval API</h1><p>Demo UI not found; use POST /search/text.</p>"


_engine = None  # lazily-loaded monitor.server.inference.InferenceEngine
_index: ServingIndex | None = None


def _load() -> tuple[object, ServingIndex]:
    """Lazily load the query encoder and the serving index from env vars."""
    global _engine, _index
    if _engine is None:
        ckpt = os.environ.get("RETRIEVAL_CHECKPOINT")
        if not ckpt:
            raise HTTPException(503, "RETRIEVAL_CHECKPOINT not set")
        from monitor.server.inference import InferenceEngine

        eng = InferenceEngine()
        eng.load(ckpt)
        _engine = eng
    if _index is None:
        path = os.environ.get("RETRIEVAL_INDEX")
        if not path:
            raise HTTPException(503, "RETRIEVAL_INDEX not set")
        backend = "faiss" if os.environ.get("RETRIEVAL_FAISS") == "1" else "numpy"
        _index = ServingIndex.load(path, backend=backend)
    return _engine, _index


class TextQuery(BaseModel):
    query: str
    k: int = 10
    bit: int | None = None


class ImageQuery(BaseModel):
    image_b64: str
    k: int = 10
    bit: int | None = None


def _code_for_bit(encoded: list[dict], bit: int) -> np.ndarray:
    """Pull the {0,1} binary code for a given bit length from engine output."""
    for entry in encoded:
        if entry["bits"] == bit:
            return np.asarray(entry["binary"], dtype=np.int8)[None, :]
    raise HTTPException(400, f"bit {bit} not produced by the model")


def _resolve_bit(index: ServingIndex, requested: int | None) -> int:
    bit = requested or index.default_bit()
    if bit not in index.per_bit:
        raise HTTPException(400, f"bit {bit} not in index (have {index.bits})")
    return bit


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "index_loaded": _index is not None,
        "items": _index.size if _index is not None else 0,
        "bits": _index.bits if _index is not None else [],
    }


@app.get("/stats")
def stats() -> dict:
    _, index = _load()
    return {
        "items": index.size,
        "bits": index.bits,
        "default_bit": index.default_bit(),
        "index_bytes": {b: int(index.per_bit[b].nbytes) for b in index.bits},
    }


@app.post("/search/text")
def search_text(q: TextQuery) -> dict:
    engine, index = _load()
    bit = _resolve_bit(index, q.bit)
    t0 = time.perf_counter()
    code = _code_for_bit(engine.encode_text(q.query), bit)
    encode_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    results = index.search(code, bit, q.k)
    search_ms = (time.perf_counter() - t1) * 1e3
    return {
        "query": q.query, "bit": bit, "k": q.k, "results": results,
        "took_ms": {"encode": round(encode_ms, 3), "search": round(search_ms, 3)},
    }


@app.post("/search/image")
def search_image(q: ImageQuery) -> dict:
    engine, index = _load()
    bit = _resolve_bit(index, q.bit)
    image = engine.decode_base64_image(q.image_b64)
    t0 = time.perf_counter()
    code = _code_for_bit(engine.encode_image(image), bit)
    encode_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    results = index.search(code, bit, q.k)
    search_ms = (time.perf_counter() - t1) * 1e3
    return {
        "bit": bit, "k": q.k, "results": results,
        "took_ms": {"encode": round(encode_ms, 3), "search": round(search_ms, 3)},
    }
