from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from monitor.server.database import (
    DB_PATH,
    clear_all_metrics,
    clear_training_metrics,
    get_eval_metrics,
    get_hash_analysis_by_id,
    get_hash_analysis_list,
    get_latest_hash_analysis,
    get_system_metrics as get_system_metrics_db,
    get_training_metrics,
    init_db,
    insert_eval_metric,
    insert_hash_analysis,
    insert_system_metric,
    insert_training_metric,
)
from monitor.server.inference import InferenceEngine
from monitor.server.models import (
    BackboneCompareRequest,
    BackboneCompareResponse,
    BackboneEncodeResponse,
    CompareRequest,
    CompareResponse,
    CompareResult,
    EncodeRequest,
    EncodeResponse,
    EvalMetric,
    HashCode,
    LoadBackboneRequest,
    LoadIndexRequest,
    LoadModelRequest,
    SearchQueryRequest,
    SearchResponse,
    SearchResult,
    SystemMetric,
    TrainingMetric,
    TrainingStatus,
)
from monitor.server.search_index import SearchIndex
from monitor.server.system_monitor import SystemMonitorThread

logger = logging.getLogger(__name__)

app = FastAPI(title="VLM Quantization Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State ---
training_status = TrainingStatus()
system_monitor = SystemMonitorThread(interval=2.0)
inference_engine = InferenceEngine()
search_index = SearchIndex()
_hash_analysis_data: dict | None = None


# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)


manager = ConnectionManager()


# --- Lifecycle ---
@app.on_event("startup")
async def startup():
    init_db()
    system_monitor.start()

    # Restore latest hash analysis from DB
    global _hash_analysis_data
    latest = get_latest_hash_analysis()
    if latest is not None:
        _hash_analysis_data = latest
        logger.info("Loaded latest hash analysis from DB")

    # Periodically broadcast system metrics via WebSocket
    async def broadcast_system():
        while True:
            metric = system_monitor.latest
            insert_system_metric(metric)
            await manager.broadcast({
                "type": "system",
                "data": metric.model_dump(),
            })
            await asyncio.sleep(2)

    asyncio.create_task(broadcast_system())


@app.on_event("shutdown")
async def shutdown():
    system_monitor.stop()


# --- WebSocket ---
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(ws)


# --- REST: Training Metrics ---
@app.post("/api/metrics/training")
async def post_training_metric(metric: TrainingMetric):
    insert_training_metric(metric)
    training_status.step = metric.step
    training_status.epoch = metric.epoch
    training_status.is_training = True
    await manager.broadcast({
        "type": "training",
        "data": metric.model_dump(),
    })
    return {"status": "ok"}


@app.post("/api/metrics/eval")
async def post_eval_metric(metric: EvalMetric):
    insert_eval_metric(metric)
    await manager.broadcast({
        "type": "eval",
        "data": metric.model_dump(),
    })
    return {"status": "ok"}


@app.get("/api/metrics/history")
async def get_metrics_history(start_step: int = 0, end_step: int | None = None):
    return {
        "training": get_training_metrics(start_step, end_step),
        "eval": get_eval_metrics(),
    }


@app.get("/api/system")
async def get_system():
    return system_monitor.latest.model_dump()


@app.get("/api/training/status")
async def get_training_status():
    return training_status.model_dump()


@app.post("/api/training/status")
async def update_training_status(status: TrainingStatus):
    # Clear previous run data when a new training session begins
    if status.is_training and status.step == 0:
        clear_training_metrics()

    training_status.epoch = status.epoch
    training_status.step = status.step
    training_status.total_epochs = status.total_epochs
    training_status.total_steps = status.total_steps
    training_status.is_training = status.is_training
    training_status.config = status.config
    await manager.broadcast({
        "type": "status",
        "data": status.model_dump(),
    })
    return {"status": "ok"}


# --- REST: Hash Analysis ---
@app.post("/api/metrics/hash_analysis")
async def post_hash_analysis(data: dict):
    global _hash_analysis_data
    _hash_analysis_data = data
    # Persist every snapshot to DB for history browsing
    try:
        insert_hash_analysis(
            epoch=data.get("epoch", 0),
            step=data.get("step", 0),
            data=data,
        )
    except Exception as e:
        logger.warning("Failed to save hash analysis to DB: %s", e)
    await manager.broadcast({"type": "hash_analysis", "data": data})
    return {"status": "ok"}


@app.get("/api/metrics/hash_analysis")
async def get_hash_analysis(id: int | None = None):
    if id is not None:
        snapshot = get_hash_analysis_by_id(id)
        return {"hash_analysis": snapshot}
    return {"hash_analysis": _hash_analysis_data}


@app.get("/api/metrics/hash_analysis/list")
async def get_hash_analysis_snapshots():
    """Return lightweight list of available snapshots (no data blobs)."""
    return {"snapshots": get_hash_analysis_list()}


@app.post("/api/metrics/reset")
async def reset_all_metrics():
    """Full reset: clear all training, eval, and hash analysis data."""
    global _hash_analysis_data
    clear_all_metrics()
    _hash_analysis_data = None
    return {"status": "ok"}


# --- REST: Inference ---
@app.post("/api/inference/load")
async def load_inference_model(req: LoadModelRequest):
    try:
        return inference_engine.load(req.checkpoint_path)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/inference/status")
async def get_inference_status():
    return inference_engine.status()


@app.get("/api/inference/checkpoints")
async def list_checkpoints(directory: str = ""):
    if not directory:
        return {"checkpoints": [], "error": "Provide directory parameter"}
    return {"checkpoints": InferenceEngine.list_checkpoints(directory)}


@app.get("/api/inference/checkpoint-info")
async def get_checkpoint_info(path: str = ""):
    """Peek at checkpoint hyperparameters without loading the full model."""
    if not path:
        return {"error": "Provide path parameter"}
    hparams = await asyncio.get_event_loop().run_in_executor(
        None, InferenceEngine.peek_hparams, path,
    )
    return {"path": path, "hparams": hparams}


@app.post("/api/inference/encode", response_model=EncodeResponse)
async def encode_input(req: EncodeRequest):
    if req.image_base64:
        image = InferenceEngine.decode_base64_image(req.image_base64)
        codes = inference_engine.encode_image(image)
    elif req.image_url:
        image = InferenceEngine.download_image(req.image_url)
        codes = inference_engine.encode_image(image)
    elif req.text:
        codes = inference_engine.encode_text(req.text)
    else:
        return {"error": "Provide image_base64, image_url, or text"}
    return EncodeResponse(codes=[HashCode(**c) for c in codes])


@app.post("/api/inference/compare", response_model=CompareResponse)
async def compare_codes(req: CompareRequest):
    codes_a = [c.model_dump() for c in req.codes_a]
    codes_b = [c.model_dump() for c in req.codes_b]
    results = InferenceEngine.compare(codes_a, codes_b)
    return CompareResponse(
        comparisons=[CompareResult(**r) for r in results]
    )


# --- REST: Backbone ---
@app.post("/api/inference/load-backbone")
async def load_backbone(req: LoadBackboneRequest):
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, inference_engine.load_backbone_only, req.model_name,
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/inference/encode-backbone", response_model=BackboneEncodeResponse)
async def encode_backbone(req: EncodeRequest):
    try:
        if req.image_base64:
            image = InferenceEngine.decode_base64_image(req.image_base64)
            embedding = inference_engine.encode_image_backbone(image)
        elif req.image_url:
            image = InferenceEngine.download_image(req.image_url)
            embedding = inference_engine.encode_image_backbone(image)
        elif req.text:
            embedding = inference_engine.encode_text_backbone(req.text)
        else:
            return {"error": "Provide image_base64, image_url, or text"}
        return BackboneEncodeResponse(embedding=embedding)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/inference/compare-backbone", response_model=BackboneCompareResponse)
async def compare_backbone(req: BackboneCompareRequest):
    result = InferenceEngine.compare_backbone(req.embedding_a, req.embedding_b)
    return BackboneCompareResponse(**result)


# --- REST: Search ---
@app.post("/api/search/load-index")
async def load_search_index(req: LoadIndexRequest):
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, search_index.load, req.index_path,
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/search/status")
async def get_search_status():
    return search_index.status


@app.post("/api/search/query", response_model=SearchResponse)
async def search_query(req: SearchQueryRequest):
    try:
        # Determine query type and encode
        if req.image_base64:
            query_type = "image"
        elif req.image_url:
            query_type = "image"
        elif req.text:
            query_type = "text"
        else:
            return {"error": "Provide image_base64, image_url, or text"}

        if req.mode == "backbone":
            # Encode query to backbone embedding
            if query_type == "image":
                image = (
                    InferenceEngine.decode_base64_image(req.image_base64)
                    if req.image_base64
                    else InferenceEngine.download_image(req.image_url)
                )
                embedding = inference_engine.encode_image_backbone(image)
            else:
                embedding = inference_engine.encode_text_backbone(req.text)

            # Cross-modal: image query → search text, text query → search image
            target_modality = "text" if query_type == "image" else "image"
            results = search_index.query_backbone(
                embedding, modality=target_modality, top_k=req.top_k,
            )
        else:
            # Hash mode — need full model loaded
            if inference_engine.model is None:
                return {"error": "Hash model not loaded (load a checkpoint first)"}

            if query_type == "image":
                image = (
                    InferenceEngine.decode_base64_image(req.image_base64)
                    if req.image_base64
                    else InferenceEngine.download_image(req.image_url)
                )
                codes = inference_engine.encode_image(image)
            else:
                codes = inference_engine.encode_text(req.text)

            # Find the right bit level
            bit_codes = None
            for c in codes:
                if c["bits"] == req.bit:
                    bit_codes = c
                    break
            if bit_codes is None:
                return {"error": f"Bit level {req.bit} not available"}

            # Convert {0,1} back to {-1,+1} for hamming_distance
            binary_signed = [1 if b == 1 else -1 for b in bit_codes["binary"]]

            target_modality = "text" if query_type == "image" else "image"
            results = search_index.query_hash(
                binary_signed, bit=req.bit, modality=target_modality, top_k=req.top_k,
            )

        return SearchResponse(
            query_type=query_type,
            mode=req.mode,
            results=[SearchResult(**r) for r in results],
        )
    except Exception as e:
        return {"error": str(e)}


# --- Static Frontend (Next.js export) ---
_frontend_out = Path(__file__).resolve().parent.parent / "frontend" / "out"

if _frontend_out.is_dir():

    @app.get("/")
    async def serve_index():
        return FileResponse(_frontend_out / "index.html")

    # Register page routes only when their HTML files exist (avoids 500
    # errors when frontend was built from an older codebase snapshot).
    _page_routes = [
        ("/inference", "inference.html"),
        ("/hash-analysis", "hash-analysis.html"),
        ("/search", "search.html"),
    ]
    for _route, _filename in _page_routes:
        _html = _frontend_out / _filename
        if _html.is_file():

            def _make_handler(p: Path):
                async def handler():
                    return FileResponse(p)
                return handler

            app.get(_route)(_make_handler(_html))

    app.mount("/", StaticFiles(directory=str(_frontend_out)), name="static")
