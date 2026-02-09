from __future__ import annotations

import torch

from src.utils.hamming import hamming_distance


def mean_average_precision(
    query_codes: torch.Tensor,
    database_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    k: int = 50,
) -> float:
    """Compute mAP@k using Hamming distance ranking.

    Args:
        query_codes: (N_q, D) in {-1, +1}
        database_codes: (N_db, D) in {-1, +1}
        query_labels: (N_q,) integer labels (image_id)
        database_labels: (N_db,) integer labels (image_id)
        k: top-k for AP computation

    Returns:
        mAP@k score.
    """
    dist = hamming_distance(query_codes, database_codes)  # (N_q, N_db)
    N_q = dist.size(0)

    # Sort by ascending distance
    _, indices = dist.sort(dim=1)
    top_k_indices = indices[:, :k]

    ap_sum = 0.0
    for i in range(N_q):
        retrieved = database_labels[top_k_indices[i]]
        relevant = (retrieved == query_labels[i]).float()

        if relevant.sum() == 0:
            continue

        # Cumulative precision
        cum_relevant = relevant.cumsum(dim=0)
        precision_at_j = cum_relevant / torch.arange(
            1, k + 1, device=relevant.device, dtype=torch.float
        )
        ap = (precision_at_j * relevant).sum() / relevant.sum()
        ap_sum += ap.item()

    return ap_sum / N_q


def precision_at_k(
    query_codes: torch.Tensor,
    database_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    k: int = 10,
) -> float:
    """Precision@k using Hamming distance ranking.

    Returns:
        Average precision@k across all queries.
    """
    dist = hamming_distance(query_codes, database_codes)
    _, indices = dist.sort(dim=1)
    top_k_indices = indices[:, :k]

    N_q = dist.size(0)
    prec_sum = 0.0
    for i in range(N_q):
        retrieved = database_labels[top_k_indices[i]]
        relevant = (retrieved == query_labels[i]).float()
        prec_sum += relevant.mean().item()

    return prec_sum / N_q


def compute_bit_entropy(binary_codes: torch.Tensor) -> torch.Tensor:
    """Compute per-bit entropy of binary codes.

    Ideal: each bit has entropy = 1.0 (50% +1, 50% -1).

    Args:
        binary_codes: (N, D) in {-1, +1}

    Returns:
        (D,) per-bit entropy values.
    """
    p_positive = (binary_codes > 0).float().mean(dim=0)
    p_positive = p_positive.clamp(1e-7, 1 - 1e-7)
    entropy = -(p_positive * p_positive.log2() + (1 - p_positive) * (1 - p_positive).log2())
    return entropy


def compute_quantization_error(
    continuous: torch.Tensor,
    binary: torch.Tensor,
) -> float:
    """Average quantization error (MSE between continuous and binary)."""
    return (continuous - binary).pow(2).mean().item()
