"""Tests for retrieval metrics (mAP, P@k, Recall@k, bit entropy)."""
from __future__ import annotations

import math

import torch

from src.utils.metrics import (
    compute_bit_entropy,
    mean_average_precision,
    precision_at_k,
    recall_at_k,
)

# 4 database items, codes at increasing Hamming distance from item 0.
DB = torch.tensor(
    [
        [1, 1, 1, 1],
        [1, 1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, -1],
    ],
    dtype=torch.float,
)
DB_LABELS = torch.tensor([0, 1, 0, 1])


def test_precision_at_k_known_values():
    q = DB[:1]  # item 0, distances to DB = [0, 1, 2, 3]
    ql = torch.tensor([0])  # relevance along ranking = [1, 0, 1, 0]
    assert math.isclose(precision_at_k(q, DB, ql, DB_LABELS, k=1), 1.0)
    assert math.isclose(precision_at_k(q, DB, ql, DB_LABELS, k=2), 0.5)
    assert math.isclose(precision_at_k(q, DB, ql, DB_LABELS, k=4), 0.5)


def test_map_known_value():
    q = DB[:1]
    ql = torch.tensor([0])  # relevance = [1, 0, 1, 0]; AP = (1.0 + 2/3) / 2
    ap = mean_average_precision(q, DB, ql, DB_LABELS, k=4)
    assert math.isclose(ap, (1.0 + 2.0 / 3.0) / 2.0, rel_tol=1e-6)


def test_recall_at_k_increases():
    q = DB[:1]
    ql = torch.tensor([1])  # relevant items appear at ranks 2 and 4
    r1 = recall_at_k(q, DB, ql, DB_LABELS, k=1)
    r2 = recall_at_k(q, DB, ql, DB_LABELS, k=2)
    r4 = recall_at_k(q, DB, ql, DB_LABELS, k=4)
    assert r1 == 0.0
    assert r2 == 1.0
    assert r1 <= r2 <= r4


def test_recall_zero_when_no_relevant():
    q = DB[:1]
    ql = torch.tensor([9])  # label absent from the database
    assert recall_at_k(q, DB, ql, DB_LABELS, k=4) == 0.0


def test_recall_monotonic_non_decreasing():
    torch.manual_seed(0)
    db = torch.randint(0, 2, (20, 8)).float() * 2 - 1
    q = torch.randint(0, 2, (6, 8)).float() * 2 - 1
    db_labels = torch.randint(0, 3, (20,))
    q_labels = torch.randint(0, 3, (6,))
    prev = -1.0
    for k in [1, 2, 5, 10, 20]:
        r = recall_at_k(q, db, q_labels, db_labels, k=k)
        assert r >= prev
        prev = r


def test_recall_multihot_overlap_path():
    q = DB[:1]
    q_labels = torch.tensor([[1.0, 0.0]])  # category 0
    db_labels = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )  # rank-1 item overlaps category 0
    assert recall_at_k(q, DB, q_labels, db_labels, k=1) == 1.0


def test_bit_entropy_balanced_is_one():
    codes = torch.tensor(
        [[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float
    )  # each bit is exactly 50% +1 / -1
    ent = compute_bit_entropy(codes)
    assert torch.allclose(ent, torch.ones(2), atol=1e-5)
