"""Tests for individual loss components and the SignSTE estimator."""
from __future__ import annotations

import torch

from src.losses.balance import BitBalanceLoss
from src.losses.eaql import EAQLLoss
from src.losses.lcs import LCSSelfDistillationLoss
from src.losses.ortho_hash import CrossModalOrthoHashLoss
from src.losses.supervised import PairwiseSupervisedLoss
from src.models.hash_layer import SignSTE


def test_signste_forward_and_backward():
    x = torch.tensor([0.5, 2.0, -3.0, -0.2], requires_grad=True)
    out = SignSTE.apply(x)
    # Forward: exact sign
    assert torch.equal(out, torch.tensor([1.0, 1.0, -1.0, -1.0]))
    # Backward: gradient passes where |x| <= 1, zeroed where |x| > 1
    out.sum().backward()
    assert torch.equal(x.grad, torch.tensor([1.0, 0.0, 0.0, 1.0]))


def test_balance_non_negative():
    torch.manual_seed(0)
    h = torch.randn(16, 8)
    loss = BitBalanceLoss(8)(h)
    assert loss.item() >= 0.0


def test_ortho_zero_for_perfect_alignment():
    # Orthonormal rows: matched pairs align (sim=1), unmatched are orthogonal (0)
    img = torch.eye(4)
    txt = torch.eye(4)
    loss = CrossModalOrthoHashLoss()(img, txt)
    assert loss.item() < 1e-6


def test_eaql_registers_ema_buffer():
    loss = EAQLLoss(ema_decay=0.9)
    cont = torch.randn(8, 16)
    assert loss._get_ema(16) is None  # lazily registered on first use
    _ = loss(cont)
    assert loss._get_ema(16) is not None
    # Second call updates EMA in place and stays finite
    assert torch.isfinite(loss(cont))


def test_lcs_zero_for_single_and_finite_for_many():
    single = [torch.randn(8, 4)]
    assert LCSSelfDistillationLoss()(single).item() == 0.0
    multi = [torch.randn(8, 4), torch.randn(8, 8), torch.randn(8, 16)]
    assert torch.isfinite(LCSSelfDistillationLoss()(multi))


def test_supervised_loss_finite():
    torch.manual_seed(0)
    img = torch.randn(6, 16)
    txt = torch.randn(6, 16)
    labels = torch.randint(0, 2, (6, 5)).float()
    loss = PairwiseSupervisedLoss(0.07)(img, txt, labels)
    assert torch.isfinite(loss)
