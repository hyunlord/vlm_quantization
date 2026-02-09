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
    lr: float


class EvalMetric(BaseModel):
    epoch: int
    map_i2t: float | None = None
    map_t2i: float | None = None
    map_i2i: float | None = None
    map_t2t: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    bit_entropy: float | None = None
    quant_error: float | None = None


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
