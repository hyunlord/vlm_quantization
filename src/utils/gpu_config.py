"""Auto-configure training parameters based on GPU capacity."""
from __future__ import annotations

import os

import torch


def _is_unified_memory(gpu_name: str, vram_gb: float) -> bool:
    """Detect unified-memory GPUs (DGX Spark GB10, Jetson, Apple-style).

    Unified memory is shared between CPU and GPU, so we cannot use all
    of it for training — the OS and CPU workloads need a share.
    """
    unified_keywords = ("gb10", "dgx spark", "grace", "jetson", "tegra")
    name_lower = gpu_name.lower()
    if any(kw in name_lower for kw in unified_keywords):
        return True
    # Heuristic: if reported VRAM > 100GB on a single GPU, likely unified
    if vram_gb > 100:
        return True
    return False


def auto_configure(
    freeze_backbone: bool = True,
    target_effective_batch: int = 256,
) -> dict:
    """Detect GPU VRAM and return optimal batch_size, accumulate_grad_batches, num_workers.

    Uses empirical per-sample memory estimates for SigLIP2 So400m + hash layers (fp16).
    Returns conservative defaults if no GPU is available.

    Handles both discrete GPUs (T4, A100, H100) and unified-memory systems
    (DGX Spark GB10) where GPU memory is shared with the OS.
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

    # Unified memory systems (DGX Spark GB10: 128GB shared CPU+GPU)
    # Reserve memory for OS + CPU workloads; cap GPU-usable portion
    unified = _is_unified_memory(gpu_name, vram_gb)
    if unified:
        # Reserve ~30GB for OS/CPU, use the rest for GPU training
        gpu_usable_gb = min(vram_gb, max(vram_gb - 30, vram_gb * 0.7))
        # Higher worker cap — ARM Grace has 20 cores, plenty of headroom
        worker_cap = 12
        print(f"  Unified memory detected: {vram_gb:.0f} GB total, "
              f"~{gpu_usable_gb:.0f} GB usable for training")
    else:
        gpu_usable_gb = vram_gb
        worker_cap = 8

    # Per-sample memory (GB, empirical for SigLIP2 + hash layers, fp16)
    # Frozen backbone: only forward activations (no backward graph stored)
    # Unfrozen backbone: forward + backward activations + gradients
    # 3-view training: 3 image forward passes + 1 text forward per sample
    # Frozen: extra views are cheap (no grad graph) → multiplier = 1.3x
    # Unfrozen: backward graph dominates → multiplier = 2.4x
    base_per_sample = 0.11 if freeze_backbone else 0.28
    view_multiplier = 1.3 if freeze_backbone else 2.4
    per_sample = base_per_sample * view_multiplier
    model_overhead = 2.0 if freeze_backbone else 4.0
    utilization = 0.65  # leave 35% headroom for peaks

    available = gpu_usable_gb * utilization - model_overhead
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

    # num_workers: scale with batch_size, cap based on system type
    num_workers = min(cpu_count, worker_cap, max(2, batch_size // 32))

    effective = batch_size * accum

    print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB{', unified' if unified else ''})")
    print(f"  Auto-config: batch_size={batch_size}, accum={accum}, "
          f"effective_batch={effective}, num_workers={num_workers}")

    return {
        "batch_size": batch_size,
        "accumulate_grad_batches": accum,
        "num_workers": num_workers,
    }
