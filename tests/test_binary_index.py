"""Tests for the binary hash search index (pack/popcount/Hamming/Matryoshka)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from src.serve.binary_index import (
    FloatIndex,
    HammingIndex,
    MatryoshkaIndex,
    has_faiss,
    pack_codes,
    to_bits01,
)


def _rand_codes(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(n, d)) * 2 - 1).astype(np.int8)


def _brute_hamming(q, db):
    """Reference Hamming distance from {-1,+1} codes (no packing)."""
    qb = (q > 0).astype(np.int32)
    dbb = (db > 0).astype(np.int32)
    return (qb[:, None, :] != dbb[None, :, :]).sum(axis=2)


def test_pack_roundtrip():
    codes = _rand_codes(5, 64)
    packed = pack_codes(codes)
    assert packed.shape == (5, 8) and packed.dtype == np.uint8
    unpacked = np.unpackbits(packed, axis=1)
    assert np.array_equal(unpacked, to_bits01(codes))


def test_hamming_search_matches_bruteforce():
    db = _rand_codes(200, 64, seed=1)
    q = _rand_codes(10, 64, seed=2)
    idx = HammingIndex.from_codes(db)
    ids, dist = idx.search(q, k=5)

    ref = _brute_hamming(q, db)
    for i in range(q.shape[0]):
        order = np.argsort(ref[i], kind="stable")[:5]
        # Distances must match the true 5 smallest (ids may differ only on ties)
        assert np.array_equal(np.sort(dist[i]), np.sort(ref[i][order]))
        assert int(dist[i, 0]) == int(ref[i].min())


def test_self_query_is_exact_match():
    db = _rand_codes(50, 32, seed=3)
    idx = HammingIndex.from_codes(db)
    ids, dist = idx.search(db, k=1)
    assert np.array_equal(ids[:, 0], np.arange(50))
    assert int(dist[:, 0].max()) == 0  # an item is identical to itself


def test_index_is_compact():
    db = _rand_codes(1000, 64, seed=4)
    hidx = HammingIndex.from_codes(db)
    fidx = FloatIndex(np.random.default_rng(0).standard_normal((1000, 1152)))
    # 64-bit codes are vastly smaller than fp32 embeddings
    assert hidx.nbytes * 100 < fidx.nbytes


def test_float_index_self_top1():
    rng = np.random.default_rng(5)
    emb = rng.standard_normal((40, 128)).astype(np.float32)
    idx = FloatIndex(emb)
    ids, sim = idx.search(emb, k=1)
    assert np.array_equal(ids[:, 0], np.arange(40))
    assert np.allclose(sim[:, 0], 1.0, atol=1e-4)


def test_matryoshka_matches_exact_long_with_enough_candidates():
    # Long code = full; short code = a prefix of it (Matryoshka property).
    full = _rand_codes(300, 64, seed=6)
    short = full[:, :16]
    mat = MatryoshkaIndex(short, full)
    exact = HammingIndex.from_codes(full)

    q_idx = np.arange(8)
    ids_m, _ = mat.search(short[q_idx], full[q_idx], k=1, candidates=300)
    ids_e, _ = exact.search(full[q_idx], k=1)
    assert np.array_equal(ids_m[:, 0], ids_e[:, 0])


def test_unpacked_query_roundtrips_to_self():
    # Reconstruct a query the way bench_search does (unpack a stored packed code).
    # np.unpackbits returns uint8, so `*2-1` must cast first or 0 underflows to 255.
    db = _rand_codes(100, 64, seed=11)
    idx = HammingIndex.from_codes(db)
    q = np.unpackbits(idx.packed[:5], axis=1)[:, :64].astype(np.int8) * 2 - 1
    ids, dist = idx.search(q, k=1)
    assert np.array_equal(ids[:, 0], np.arange(5))  # each query finds itself
    assert int(dist[:, 0].max()) == 0


@pytest.mark.skipif(not has_faiss(), reason="faiss-cpu not installed")
def test_faiss_backend_matches_numpy():
    if "torch" in sys.modules:
        # faiss + torch in one process can deadlock on macOS (duplicate libomp);
        # run this file on its own to exercise the faiss backend.
        pytest.skip("torch already imported in-process; run test_binary_index.py alone")
    db = _rand_codes(500, 64, seed=7)
    q = _rand_codes(16, 64, seed=8)
    a = HammingIndex.from_codes(db, backend="numpy").search(q, k=5)[1]
    b = HammingIndex.from_codes(db, backend="faiss").search(q, k=5)[1]
    assert np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1))
