"""Multiscale / UNet-style hash heads for the head-structure gate.

All heads share the baseline interface: input (B, input_dim) -> list[dict] per scale,
each dict = {"continuous": tanh(L2(BN(raw_s))), "binary": signSTE(L2(BN(raw_s)))}.

CRITICAL identity constraint: each scale's code is produced from its OWN raw_s vector and
is independently usable for Hamming retrieval (no need to concatenate other scales).
The variants differ only in HOW raw_s is produced:

  baseline (NestedHashLayer)  : single shared MLP -> 1024-d, raw_s = prefix[:s]   (depth-1)
  ConvHead / ConvUNetHead     : 1D-conv over the embedding axis (Gate 1, locality test)
  DepthHead(mode=depth|skip|topdown) : coarse scales from shallow layers, fine from deep;
                                skip = UNet-style cross-scale exchange (Gate 3 ★)
  ResidualHead                : coarse-to-fine residual quantization in feature space (Gate 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hash_layer import SignSTE

DEFAULT_BITS = [8, 16, 32, 64, 128, 256, 512, 1024]


class ScaleTail(nn.Module):
    """Per-scale BatchNorm -> L2 -> {tanh continuous, signSTE binary}. Shared by all heads."""

    def __init__(self, bit_list):
        super().__init__()
        self.bit_list = sorted(bit_list)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(b) for b in self.bit_list])

    def one(self, raw_s, idx):
        z = F.normalize(self.batch_norms[idx](raw_s), p=2, dim=1)
        return {"continuous": torch.tanh(z), "binary": SignSTE.apply(z)}


# ───────────────────────── Gate 1: conv heads ─────────────────────────
class ConvHead(nn.Module):
    """Treat the embedding (input_dim,) as a length-input_dim 1-channel sequence; a 1D-conv
    stack extracts features, then a linear maps to max_bit; prefix-slice -> per-scale tail.
    Tests whether the embedding axis has any 1D locality conv can exploit (expected: no)."""

    def __init__(self, input_dim, hidden_dim=384, bit_list=None, dropout=0.1):
        super().__init__()
        bit_list = sorted(bit_list or DEFAULT_BITS)
        self.bit_list = bit_list; self.max_bit = bit_list[-1]
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1), nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.max_bit))
        self.tail = ScaleTail(bit_list)
        nn.init.xavier_uniform_(self.proj[-1].weight, gain=0.1); nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        f = self.conv(x.unsqueeze(1)).flatten(1)
        raw = self.proj(f)
        return [self.tail.one(raw[:, :b], i) for i, b in enumerate(self.bit_list)]


class ConvUNetHead(nn.Module):
    """1D conv-UNet (encoder-decoder + skip) over the embedding axis -> max_bit. Gate 1 variant
    that gives conv its best shot via multi-resolution skips along the dimension axis."""

    def __init__(self, input_dim, hidden_dim=384, bit_list=None, dropout=0.1):
        super().__init__()
        bit_list = sorted(bit_list or DEFAULT_BITS)
        self.bit_list = bit_list; self.max_bit = bit_list[-1]
        self.e1 = nn.Sequential(nn.Conv1d(1, 16, 5, 2, 2), nn.GELU())     # L/2
        self.e2 = nn.Sequential(nn.Conv1d(16, 32, 5, 2, 2), nn.GELU())    # L/4
        self.mid = nn.Sequential(nn.Conv1d(32, 32, 3, 1, 1), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv1d(32 + 16, 16, 3, 1, 1), nn.GELU())  # up(mid)=32ch + skip e1=16ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.proj = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.max_bit))
        self.tail = ScaleTail(bit_list)
        nn.init.xavier_uniform_(self.proj[-1].weight, gain=0.1); nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        s = x.unsqueeze(1)
        e1 = self.e1(s); e2 = self.e2(e1); m = self.mid(e2)
        up = self.up(m)
        if up.shape[-1] != e1.shape[-1]:
            up = F.pad(up, (0, e1.shape[-1] - up.shape[-1]))
        d = self.d2(torch.cat([up, e1], dim=1))
        raw = self.proj(d.flatten(1))
        return [self.tail.one(raw[:, :b], i) for i, b in enumerate(self.bit_list)]


# ───────────────────────── Gate 3 ★: depth-wise extraction ─────────────────────────
class DepthHead(nn.Module):
    """Coarse codes from shallow trunk layers, fine codes from deep layers (UNet-style depth->scale).
    mode:
      'depth'   : each scale's raw read only from its assigned depth's features (depth-only, no exchange)
      'skip'    : fine-scale heads also receive the shallowest features (UNet skip = cross-scale exchange)
      'topdown' : coarse continuous code conditions the fine-scale head (coarse->fine top-down)
    Each scale still produces an independent raw_s -> independently searchable.
    """

    def __init__(self, input_dim, hidden_dim=384, bit_list=None, dropout=0.1, mode="skip", n_depth=4):
        super().__init__()
        bit_list = sorted(bit_list or DEFAULT_BITS)
        self.bit_list = bit_list; self.mode = mode
        # assign scales to depths, coarse->shallow ... fine->deep
        n = len(bit_list); per = max(1, n // n_depth)
        self.depth_of = {}
        for i, b in enumerate(bit_list):
            self.depth_of[b] = min(n_depth - 1, i // per)
        self.n_depth = n_depth
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(n_depth)])
        # per-scale projection; input width depends on mode
        self.proj = nn.ModuleDict()
        for b in bit_list:
            in_w = hidden_dim
            if mode == "skip" and self.depth_of[b] > 0:
                in_w = hidden_dim * 2                      # deep feats + shallow skip
            if mode == "topdown" and i_prev_exists(bit_list, b):
                in_w = hidden_dim + prev_scale(bit_list, b)  # deep feats + coarser continuous code
            self.proj[str(b)] = nn.Linear(in_w, b)
            nn.init.xavier_uniform_(self.proj[str(b)].weight, gain=0.1); nn.init.zeros_(self.proj[str(b)].bias)
        self.tail = ScaleTail(bit_list)

    def forward(self, x):
        h = self.stem(x)
        feats = []
        cur = h
        for blk in self.blocks:
            cur = blk(cur); feats.append(cur)
        shallow = feats[0]
        outs = []; prev_cont = None
        for i, b in enumerate(self.bit_list):
            fg = feats[self.depth_of[b]]
            if self.mode == "skip" and self.depth_of[b] > 0:
                inp = torch.cat([fg, shallow], dim=1)
            elif self.mode == "topdown" and prev_cont is not None:
                inp = torch.cat([fg, prev_cont], dim=1)
            else:
                inp = fg
            raw = self.proj[str(b)](inp)
            out = self.tail.one(raw, i)
            prev_cont = out["continuous"]
            outs.append(out)
        return outs


def i_prev_exists(bit_list, b):
    return sorted(bit_list).index(b) > 0


def prev_scale(bit_list, b):
    bl = sorted(bit_list); return bl[bl.index(b) - 1]


# ───────────────────────── Gate 2: coarse-to-fine residual ─────────────────────────
class ResidualHead(nn.Module):
    """Coarse-to-fine residual quantization in feature space: scale s_k encodes the residual
    left by all coarser scales. Each scale's raw is still an independent code.
    resid_0 = stem(x); for each scale: raw_s = enc_s(resid); resid <- resid - dec_s(raw_s)."""

    def __init__(self, input_dim, hidden_dim=384, bit_list=None, dropout=0.1):
        super().__init__()
        bit_list = sorted(bit_list or DEFAULT_BITS)
        self.bit_list = bit_list
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout))
        self.enc = nn.ModuleList([nn.Linear(hidden_dim, b) for b in bit_list])
        self.dec = nn.ModuleList([nn.Linear(b, hidden_dim) for b in bit_list])
        for e in self.enc:
            nn.init.xavier_uniform_(e.weight, gain=0.1); nn.init.zeros_(e.bias)
        self.tail = ScaleTail(bit_list)

    def forward(self, x):
        resid = self.stem(x)
        outs = []
        for i, b in enumerate(self.bit_list):
            raw = self.enc[i](resid)
            outs.append(self.tail.one(raw, i))
            resid = resid - self.dec[i](raw)          # remove what this scale captured
        return outs


def build_head(variant, input_dim=1152, hidden_dim=384, bit_list=None, dropout=0.1):
    bit_list = bit_list or DEFAULT_BITS
    if variant == "baseline":
        from src.models.nested_hash_layer import NestedHashLayer
        return NestedHashLayer(input_dim, hidden_dim, bit_list, dropout)
    if variant == "conv":
        return ConvHead(input_dim, hidden_dim, bit_list, dropout)
    if variant == "convunet":
        return ConvUNetHead(input_dim, hidden_dim, bit_list, dropout)
    if variant == "depth":
        return DepthHead(input_dim, hidden_dim, bit_list, dropout, mode="depth")
    if variant == "depth_skip":
        return DepthHead(input_dim, hidden_dim, bit_list, dropout, mode="skip")
    if variant == "depth_topdown":
        return DepthHead(input_dim, hidden_dim, bit_list, dropout, mode="topdown")
    if variant == "residual":
        return ResidualHead(input_dim, hidden_dim, bit_list, dropout)
    raise ValueError(variant)
