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


def xor_hamming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamming distance via XOR for {0, 1} codes.

    Args:
        a: (N, D) binary codes in {0, 1} (uint8)
        b: (M, D) binary codes in {0, 1} (uint8)

    Returns:
        (N, M) integer Hamming distance matrix.
    """
    # (N, 1, D) XOR (1, M, D) -> (N, M, D) -> sum -> (N, M)
    return (a.unsqueeze(1) ^ b.unsqueeze(0)).sum(dim=-1).long()


def pack_binary_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack {0, 1} binary codes into uint8 for storage efficiency.

    64 bits -> 8 bytes, 128 bits -> 16 bytes per code.

    Args:
        codes: (N, D) binary codes in {0, 1}, D must be multiple of 8.

    Returns:
        (N, D // 8) packed uint8 tensor.
    """
    N, D = codes.shape
    assert D % 8 == 0, f"Hash dim must be multiple of 8, got {D}"
    codes = codes.to(torch.uint8).view(N, D // 8, 8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)
    return (codes * powers).sum(dim=-1).to(torch.uint8)
