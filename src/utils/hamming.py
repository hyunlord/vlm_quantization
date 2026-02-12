from __future__ import annotations

import torch


def to_binary_01(codes: torch.Tensor) -> torch.Tensor:
    """Convert {-1, +1} codes to {0, 1}."""
    return (codes > 0).to(torch.uint8)


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamming distance between two sets of {-1, +1} binary codes.

    hamming(a, b) = (D - a @ b.T) / 2

    Args:
        a: (N, D) binary codes in {-1, +1}
        b: (M, D) binary codes in {-1, +1}

    Returns:
        (N, M) integer Hamming distance matrix.
    """
    D = a.size(1)
    dot = a.float() @ b.float().t()
    return ((D - dot) / 2).long()


