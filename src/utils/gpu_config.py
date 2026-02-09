"""Auto-configure training parameters based on GPU capacity."""
from __future__ import annotations

import os

import torch


def auto_configure(
    freeze_backbone: bool = True,
    target_effective_batch: int = 256,
) -> dict:
    """Detect GPU VRAM and return optimal batch_size, accumulate_grad_batches, num_workers.

    Uses empirical per-sample memory estimates for SigLIP2 So400m + hash layers (fp16).
    Returns conservative defaults if no GPU is available.
    """
    if not torch.cuda.is_available():
        return {
            "batch_size": 32,
            "accumulate_grad_batches": 8,
            "num_workers": 2,
        }

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    gpu_name = props.name
    cpu_count = os.cpu_count() or 4

    # Per-sample memory (GB, empirical for SigLIP2 + hash layers, fp16)
    # Frozen backbone: only forward activations
    # Unfrozen backbone: forward + backward activations + gradients
    per_sample = 0.11 if freeze_backbone else 0.28
    model_overhead = 2.0 if freeze_backbone else 4.0
    utilization = 0.65  # leave 35% headroom for peaks

    available = vram_gb * utilization - model_overhead
    optimal_batch = int(available / per_sample)
    # Round down to multiple of 32 for GPU efficiency, min 32
    optimal_batch = max(32, (optimal_batch // 32) * 32)

    # Determine accumulate_grad_batches to reach target effective batch
    if optimal_batch >= target_effective_batch:
        accum = 1
        batch_size = optimal_batch
    else:
        batch_size = optimal_batch
        accum = max(1, -(-target_effective_batch // batch_size))  # ceil div

    # num_workers: scale with batch_size, cap at CPU count and 16
    num_workers = min(cpu_count, 16, max(4, batch_size // 16))

    effective = batch_size * accum

    print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"  Auto-config: batch_size={batch_size}, accum={accum}, "
          f"effective_batch={effective}, num_workers={num_workers}")

    return {
        "batch_size": batch_size,
        "accumulate_grad_batches": accum,
        "num_workers": num_workers,
    }
