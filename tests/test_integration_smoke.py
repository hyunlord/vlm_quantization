"""Backbone-free integration smoke test for the trainable hashing path.

Exercises NestedHashLayer -> CombinedHashLoss -> backward -> optimizer step
end-to-end on CPU with random embeddings (no SigLIP2 download required),
verifying gradients flow and the learnable objective actually decreases.
"""
from __future__ import annotations

import torch

from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer


def _paired_embeddings(batch=16, dim=64, seed=0):
    """Image/text embeddings sharing a latent block so contrastive is learnable."""
    torch.manual_seed(seed)
    shared = torch.randn(batch, dim // 2)
    img = torch.cat([shared, torch.randn(batch, dim // 2)], dim=1)
    txt = torch.cat([shared, torch.randn(batch, dim // 2)], dim=1)
    return img, txt


def test_hash_pipeline_trains_and_flows_gradients():
    bits = [8, 16, 32]
    img_layer = NestedHashLayer(64, 32, bits, 0.0)
    txt_layer = NestedHashLayer(64, 32, bits, 0.0)
    loss_fn = CombinedHashLoss(bits)
    img_layer.train()
    txt_layer.train()
    loss_fn.train()

    img_emb, txt_emb = _paired_embeddings()
    params = list(img_layer.parameters()) + list(txt_layer.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)

    first_contrastive = None
    last_contrastive = None
    for _ in range(40):
        opt.zero_grad()
        out = loss_fn(img_layer(img_emb), txt_layer(txt_emb))
        total = out["total"]
        assert torch.isfinite(total)
        total.backward()
        # Gradients must reach the first projection weight of both heads
        g_img = img_layer.hash_head[0].weight.grad
        g_txt = txt_layer.hash_head[0].weight.grad
        assert g_img is not None and torch.isfinite(g_img).all()
        assert g_txt is not None and torch.isfinite(g_txt).all()
        opt.step()
        if first_contrastive is None:
            first_contrastive = out["contrastive"].item()
        last_contrastive = out["contrastive"].item()

    # The learnable cross-modal contrastive term should drop with shared latent
    assert last_contrastive < first_contrastive


def test_aux_text_path_runs_end_to_end():
    bits = [8, 16]
    img_layer = NestedHashLayer(64, 32, bits, 0.0).train()
    txt_layer = NestedHashLayer(64, 32, bits, 0.0).train()
    loss_fn = CombinedHashLoss(bits, focal_gamma=2.0)

    img_emb, txt_emb = _paired_embeddings(seed=1)
    _, aux_emb = _paired_embeddings(seed=2)

    out = loss_fn(
        img_layer(img_emb),
        txt_layer(txt_emb),
        aux_text_outputs=txt_layer(aux_emb),
    )
    out["total"].backward()
    assert torch.isfinite(out["total"])
    assert img_layer.hash_head[0].weight.grad is not None
