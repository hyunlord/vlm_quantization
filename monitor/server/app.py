from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from monitor.server.database import (
    clear_all_metrics,
    clear_training_metrics,
    get_eval_metrics,
    get_system_metrics as get_system_metrics_db,
    get_training_metrics,
    init_db,
    insert_eval_metric,
    insert_system_metric,
    insert_training_metric,
)
from monitor.server.inference import InferenceEngine
from monitor.server.models import (
    CompareRequest,
    CompareResponse,
    CompareResult,
    EncodeRequest,
    EncodeResponse,
    EvalMetric,
    HashCode,
    LoadModelRequest,
    SystemMetric,
    TrainingMetric,
    TrainingStatus,
)
from monitor.server.system_monitor import SystemMonitorThread

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
    await manager.broadcast({"type": "hash_analysis", "data": data})
    return {"status": "ok"}


@app.get("/api/metrics/hash_analysis")
async def get_hash_analysis():
    return {"hash_analysis": _hash_analysis_data}


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


# --- Static Frontend (Next.js export) ---
_frontend_out = Path(__file__).resolve().parent.parent / "frontend" / "out"

if _frontend_out.is_dir():

    @app.get("/")
    async def serve_index():
        return FileResponse(_frontend_out / "index.html")

    @app.get("/inference")
    async def serve_inference():
        return FileResponse(_frontend_out / "inference.html")

    @app.get("/hash-analysis")
    async def serve_hash_analysis():
        return FileResponse(_frontend_out / "hash-analysis.html")

    app.mount("/", StaticFiles(directory=str(_frontend_out)), name="static")
