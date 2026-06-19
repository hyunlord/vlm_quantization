"""FastAPI server for the browser-side 1-bit search demo (web/ v1).

Architecture is INVERTED vs demo/live_server.py: the server does NOT search. It only
  (a) encodes a short query text into a 1024-bit code, and
  (b) serves static files (frontend + index.bin + meta.json + thumbnails).
All Hamming search happens client-side in static/app.js. There is no faiss / vector
DB / corpus on the query path here — see verify_parity.py for the proof that the
client search reproduces faiss.

Endpoints:
  POST /encode_query  {"text": "..."} -> {"code": "<base64 of 128 bytes>", "bits", "encode_ms"}
  GET  /              -> static/index.html
  GET  /app.js                        (static)
  GET  /data/index.bin, /data/meta.json, /data/index_info.json   (static)
  GET  /thumbs/<id>.jpg               (static)

Run on DGX (where the checkpoint lives):
  cd ~/github/vlm_quantization
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/uvicorn web.query_server:app --host 0.0.0.0 --port 8300
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.common import CODE_BITS, CODE_BYTES, HEAD_PATH, Encoder

STATIC = Path(__file__).resolve().parent / "static"
(STATIC / "data").mkdir(parents=True, exist_ok=True)
(STATIC / "thumbs").mkdir(parents=True, exist_ok=True)

print(f"[web] loading encoder head: {HEAD_PATH}", flush=True)
enc = Encoder()
try:  # warm the text tower so the first real query isn't a cold outlier
    enc.text_code_bytes("warmup query")
    print(f"[web] encoder ready (bits={CODE_BITS}, norm_in={enc.norm_in})", flush=True)
except Exception as e:  # pragma: no cover
    print(f"[web] warmup skipped: {e}", flush=True)

app = FastAPI(title="Browser-side 1-bit search — encode-only server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class Query(BaseModel):
    text: str


@app.post("/encode_query")
def encode_query(q: Query):
    """Encode text -> 1024-bit code (128 bytes), base64. No search is performed."""
    t0 = time.perf_counter()
    code = enc.text_code_bytes(q.text)  # exactly CODE_BYTES bytes, same packing as index
    encode_ms = (time.perf_counter() - t0) * 1e3
    return {"code": base64.b64encode(code).decode("ascii"),
            "bits": CODE_BITS, "bytes": CODE_BYTES, "encode_ms": round(encode_ms, 2)}


@app.get("/", response_class=HTMLResponse)
def index():
    f = STATIC / "index.html"
    if not f.exists():
        return "<h1>web/static/index.html missing</h1>"
    return f.read_text(encoding="utf-8")


@app.get("/app.js")
def app_js():
    return FileResponse(STATIC / "app.js", media_type="application/javascript")


# Static artifacts (index.bin / meta.json / index_info.json and thumbnails).
app.mount("/data", StaticFiles(directory=str(STATIC / "data")), name="data")
app.mount("/thumbs", StaticFiles(directory=str(STATIC / "thumbs")), name="thumbs")
