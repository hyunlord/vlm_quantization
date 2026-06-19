"""Tests for the NestedHashLayer (prefix-nested multi-resolution hashing)."""
from __future__ import annotations

import torch

from src.models.nested_hash_layer import NestedHashLayer


def test_output_count_and_shapes():
    bits = [8, 16, 32]
    layer = NestedHashLayer(64, 32, bits, 0.0).eval()
    out = layer(torch.randn(5, 64))
    assert len(out) == len(bits)
    for o, b in zip(out, bits):
        assert o["continuous"].shape == (5, b)
        assert o["binary"].shape == (5, b)


def test_continuous_and_binary_ranges():
    layer = NestedHashLayer(32, 16, [8, 16], 0.0).eval()
    out = layer(torch.randn(4, 32))
    for o in out:
        c, b = o["continuous"], o["binary"]
        assert c.max().item() <= 1.0 and c.min().item() >= -1.0
        assert torch.equal(b.abs(), torch.ones_like(b))


def test_prefix_property_in_projection():
    # Shorter codes are strict prefixes of longer ones at the projection level.
    layer = NestedHashLayer(32, 16, [8, 16, 32], 0.0).eval()
    raw = layer.hash_head(torch.randn(4, 32))
    assert torch.equal(raw[:, :8], raw[:, :16][:, :8])
    assert torch.equal(raw[:, :16], raw[:, :32][:, :16])


def test_bit_list_is_sorted():
    layer = NestedHashLayer(16, 8, [32, 8, 16], 0.0)
    assert layer.bit_list == [8, 16, 32]
