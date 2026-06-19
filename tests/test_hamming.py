"""Tests for Hamming distance utilities."""
from __future__ import annotations

import torch

from src.utils.hamming import hamming_distance, to_binary_01


def test_to_binary_01_maps_signs():
    codes = torch.tensor([[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]])
    out = to_binary_01(codes)
    assert out.dtype == torch.uint8
    assert torch.equal(
        out, torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.uint8)
    )


def test_hamming_identical_is_zero():
    a = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    assert hamming_distance(a, a).item() == 0


def test_hamming_opposite_is_full():
    a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    b = -a
    assert hamming_distance(a, b).item() == 4


def test_hamming_matches_bruteforce():
    torch.manual_seed(0)
    a = torch.randint(0, 2, (5, 16)).float() * 2 - 1
    b = torch.randint(0, 2, (7, 16)).float() * 2 - 1
    dist = hamming_distance(a, b)

    brute = torch.zeros(5, 7, dtype=torch.long)
    for i in range(5):
        for j in range(7):
            brute[i, j] = (a[i] != b[j]).sum()

    assert torch.equal(dist, brute)


def test_hamming_output_shape():
    a = torch.ones(3, 8)
    b = torch.ones(4, 8)
    assert hamming_distance(a, b).shape == (3, 4)
