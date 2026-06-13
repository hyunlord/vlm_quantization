from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.hamming import hamming_distance


def _relevance_matrix(
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    top_indices: torch.Tensor,
) -> torch.Tensor:
    """Relevance matrix (Nq, k) in {0, 1} for gathered top-k database indices.

    Supports both:
        - 1D integer labels (image_id matching): relevant if equal
        - 2D multi-hot labels (category overlap): relevant if any shared category
    """
    if query_labels.dim() == 1:
        retrieved = database_labels[top_indices]  # (Nq, k)
        return (retrieved == query_labels.unsqueeze(1)).float()
    # Multi-hot: relevant if the label vectors share at least one category.
    db = database_labels[top_indices].float()  # (Nq, k, C)
    q = query_labels.unsqueeze(1).float()  # (Nq, 1, C)
    return ((db * q).sum(dim=-1) > 0).float()


def _query_chunk(query_labels: torch.Tensor) -> int:
    """Pick a query-chunk size that bounds peak memory of the relevance matrix.

    Multi-hot labels materialize a (chunk, k, C) tensor, so use a smaller chunk.
    """
    return 256 if query_labels.dim() > 1 else 2048


def _ap_per_query(rel: torch.Tensor) -> torch.Tensor:
    """Average precision per query from a (Nq, k) relevance matrix.

    Queries with no relevant item yield 0 (and are still counted in the mean).
    """
    k = rel.size(1)
    cum = rel.cumsum(dim=1)
    ranks = torch.arange(1, k + 1, device=rel.device, dtype=torch.float)
    precision_at_j = cum / ranks
    num_rel = rel.sum(dim=1)
    ap = (precision_at_j * rel).sum(dim=1) / num_rel.clamp(min=1.0)
    return torch.where(num_rel > 0, ap, torch.zeros_like(ap))


def _ranked_top_k(
    scores: torch.Tensor, k: int, descending: bool
) -> tuple[torch.Tensor, int]:
    """Return top-k indices (Nq, k) ranked by score, plus the effective k."""
    actual_k = min(k, scores.size(1))
    _, indices = scores.sort(dim=1, descending=descending)
    return indices[:, :actual_k], actual_k


def _map_from_ranking(
    top_indices: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
) -> float:
    """Vectorized mAP from a precomputed (Nq, k) ranking, chunked over queries."""
    N_q = top_indices.size(0)
    chunk = _query_chunk(query_labels)
    ap_sum = 0.0
    for s in range(0, N_q, chunk):
        e = min(s + chunk, N_q)
        rel = _relevance_matrix(
            query_labels[s:e], database_labels, top_indices[s:e]
        )
        ap_sum += _ap_per_query(rel).sum().item()
    return ap_sum / N_q


def _precision_from_ranking(
    top_indices: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
) -> float:
    """Vectorized mean precision@k from a precomputed (Nq, k) ranking."""
    N_q = top_indices.size(0)
    chunk = _query_chunk(query_labels)
    prec_sum = 0.0
    for s in range(0, N_q, chunk):
        e = min(s + chunk, N_q)
        rel = _relevance_matrix(
            query_labels[s:e], database_labels, top_indices[s:e]
        )
        prec_sum += rel.mean(dim=1).sum().item()
    return prec_sum / N_q


def cosine_mean_average_precision(
    query_emb: torch.Tensor,
    database_emb: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    k: int = 5000,
) -> float:
    """Compute mAP@k using cosine similarity ranking (backbone baseline).

    Args:
        query_emb: (N_q, D) continuous embeddings.
        database_emb: (N_db, D) continuous embeddings.
        query_labels: (N_q,) integer or (N_q, C) multi-hot labels.
        database_labels: (N_db,) integer or (N_db, C) multi-hot labels.
        k: top-k for AP computation.

    Returns:
        mAP@k score.
    """
    query_norm = F.normalize(query_emb, dim=1)
    db_norm = F.normalize(database_emb, dim=1)
    sim = query_norm @ db_norm.T  # (N_q, N_db)
    top_indices, _ = _ranked_top_k(sim, k, descending=True)
    return _map_from_ranking(top_indices, query_labels, database_labels)


def mean_average_precision(
    query_codes: torch.Tensor,
    database_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    k: int = 5000,
) -> float:
    """Compute mAP@k using Hamming distance ranking.

    Args:
        query_codes: (N_q, D) in {-1, +1}
        database_codes: (N_db, D) in {-1, +1}
        query_labels: (N_q,) integer or (N_q, C) multi-hot labels
        database_labels: (N_db,) integer or (N_db, C) multi-hot labels
        k: top-k for AP computation

    Returns:
        mAP@k score.
    """
    dist = hamming_distance(query_codes, database_codes)  # (N_q, N_db)
    top_indices, _ = _ranked_top_k(dist, k, descending=False)
    return _map_from_ranking(top_indices, query_labels, database_labels)


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
    top_indices, _ = _ranked_top_k(dist, k, descending=False)
    return _precision_from_ranking(top_indices, query_labels, database_labels)


def recall_at_k(
    query_codes: torch.Tensor,
    database_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    k: int = 10,
) -> float:
    """Recall@k using Hamming distance ranking.

    Recall@k = fraction of queries for which at least one relevant item
    appears within the top-k retrieved results. Monotonically non-decreasing
    in k. Queries with no relevant item in the database contribute 0.

    Args:
        query_codes: (N_q, D) in {-1, +1}
        database_codes: (N_db, D) in {-1, +1}
        query_labels: (N_q,) integer or (N_q, C) multi-hot labels
        database_labels: (N_db,) integer or (N_db, C) multi-hot labels
        k: cutoff rank.

    Returns:
        Average recall@k across all queries.
    """
    dist = hamming_distance(query_codes, database_codes)
    top_indices, _ = _ranked_top_k(dist, k, descending=False)

    N_q = top_indices.size(0)
    chunk = _query_chunk(query_labels)
    hit_sum = 0.0
    for s in range(0, N_q, chunk):
        e = min(s + chunk, N_q)
        rel = _relevance_matrix(
            query_labels[s:e], database_labels, top_indices[s:e]
        )
        hit_sum += (rel.sum(dim=1) > 0).float().sum().item()
    return hit_sum / N_q


def cosine_precision_at_k(
    query_emb: torch.Tensor,
    database_emb: torch.Tensor,
    query_labels: torch.Tensor,
    database_labels: torch.Tensor,
    k: int = 10,
) -> float:
    """Precision@k using cosine similarity ranking (backbone baseline)."""
    query_norm = F.normalize(query_emb, dim=1)
    db_norm = F.normalize(database_emb, dim=1)
    sim = query_norm @ db_norm.T
    top_indices, _ = _ranked_top_k(sim, k, descending=True)
    return _precision_from_ranking(top_indices, query_labels, database_labels)


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
