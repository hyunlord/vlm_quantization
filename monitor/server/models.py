from __future__ import annotations

from pydantic import BaseModel


class TrainingMetric(BaseModel):
    step: int
    epoch: int
    loss_total: float
    loss_contrastive: float
    loss_quantization: float
    loss_balance: float
    loss_consistency: float
    loss_ortho: float = 0.0
    loss_lcs: float = 0.0
    lr: float


class EvalMetric(BaseModel):
    epoch: int
    step: int | None = None
    map_i2t: float | None = None
    map_t2i: float | None = None
    map_i2i: float | None = None
    map_t2t: float | None = None
    backbone_map_i2t: float | None = None
    backbone_map_t2i: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    backbone_p1: float | None = None
    backbone_p5: float | None = None
    backbone_p10: float | None = None
    bit_entropy: float | None = None
    quant_error: float | None = None
    # Per-bit quality metrics
    bit_entropy_16: float | None = None
    bit_entropy_32: float | None = None
    bit_entropy_64: float | None = None
    bit_entropy_128: float | None = None
    quant_error_16: float | None = None
    quant_error_32: float | None = None
    quant_error_64: float | None = None
    quant_error_128: float | None = None
    # Validation losses (for train vs val comparison)
    val_loss_total: float | None = None
    val_loss_contrastive: float | None = None
    val_loss_quantization: float | None = None
    val_loss_balance: float | None = None
    val_loss_consistency: float | None = None
    val_loss_ortho: float | None = None
    val_loss_lcs: float | None = None


class SystemMetric(BaseModel):
    gpu_util: float = 0.0
    gpu_mem_used: float = 0.0
    gpu_mem_total: float = 0.0
    gpu_temp: float = 0.0
    gpu_name: str = ""
    cpu_util: float = 0.0
    ram_used: float = 0.0
    ram_total: float = 0.0


class TrainingStatus(BaseModel):
    epoch: int = 0
    step: int = 0
    total_epochs: int = 0
    total_steps: int = 0
    is_training: bool = False
    config: dict | None = None


# --- Inference models ---

class LoadModelRequest(BaseModel):
    checkpoint_path: str


class EncodeRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    text: str | None = None


class HashCode(BaseModel):
    bits: int
    binary: list[int]
    continuous: list[float]


class EncodeResponse(BaseModel):
    codes: list[HashCode]


class CompareRequest(BaseModel):
    codes_a: list[HashCode]
    codes_b: list[HashCode]


class CompareResult(BaseModel):
    bits: int
    hamming: int
    max_distance: int
    similarity: float


class CompareResponse(BaseModel):
    comparisons: list[CompareResult]


# --- Backbone models ---

class LoadBackboneRequest(BaseModel):
    model_name: str = "google/siglip2-so400m-patch14-384"


class BackboneEncodeResponse(BaseModel):
    embedding: list[float]


class BackboneCompareRequest(BaseModel):
    embedding_a: list[float]
    embedding_b: list[float]


class BackboneCompareResponse(BaseModel):
    cosine_similarity: float


# --- Search models ---

class SearchQueryRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    text: str | None = None
    mode: str = "hash"
    bit: int = 64
    top_k: int = 20


class LoadIndexRequest(BaseModel):
    index_path: str


class SearchResult(BaseModel):
    rank: int
    image_id: int
    caption: str
    thumbnail: str
    score: float
    distance: int | None = None


class SearchResponse(BaseModel):
    query_type: str
    mode: str
    results: list[SearchResult]
