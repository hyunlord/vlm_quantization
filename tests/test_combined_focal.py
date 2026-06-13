"""Tests for focal InfoNCE and the CombinedHashLoss orchestrator."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.combined import CombinedHashLoss
from src.losses.contrastive import CrossModalContrastiveLoss


def _outputs(bit_list, seed=0, batch=4):
    torch.manual_seed(seed)
    return [
        {
            "continuous": torch.randn(batch, b),
            "binary": torch.randn(batch, b).sign(),
        }
        for b in bit_list
    ]


def test_focal_gamma_zero_equals_plain_infonce():
    torch.manual_seed(0)
    a, b = torch.randn(8, 16), torch.randn(8, 16)

    loss = CrossModalContrastiveLoss(0.07, focal_gamma=0.0)(a, b)

    # Reference: standard symmetric InfoNCE
    an, bn = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = an @ bn.t() / 0.07
    labels = torch.arange(8)
    ref = (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    ) / 2.0
    assert torch.allclose(loss, ref)


def test_focal_gamma_nonzero_changes_loss():
    torch.manual_seed(1)
    a, b = torch.randn(8, 16), torch.randn(8, 16)
    plain = CrossModalContrastiveLoss(0.07, focal_gamma=0.0)(a, b)
    focal = CrossModalContrastiveLoss(0.07, focal_gamma=2.0)(a, b)
    assert not torch.allclose(plain, focal)
    assert focal.item() >= 0.0


def test_combined_loss_finite_default():
    bits = [8, 16, 32]
    loss = CombinedHashLoss(bits)(_outputs(bits, seed=0), _outputs(bits, seed=1))
    assert torch.isfinite(loss["total"])


def test_combined_aux_text_changes_contrastive():
    bits = [8, 16]
    img, txt, aux = (
        _outputs(bits, seed=0),
        _outputs(bits, seed=1),
        _outputs(bits, seed=2),
    )
    base = CombinedHashLoss(bits)(img, txt)
    with_aux = CombinedHashLoss(bits)(img, txt, aux_text_outputs=aux)
    assert torch.isfinite(with_aux["total"])
    assert not torch.allclose(base["contrastive"], with_aux["contrastive"])
