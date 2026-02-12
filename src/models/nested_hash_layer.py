from __future__ import annotations

import torch
import torch.nn as nn

from src.models.hash_layer import SignSTE


class NestedHashLayer(nn.Module):
    """Multi-resolution hash layer with prefix slicing or per-bit heads.

    Produces hash codes at multiple bit lengths.

    Two modes (controlled by `progressive` flag):
        - prefix slicing (progressive=False, default for backward compat):
            input_dim -> hidden_dim -> max_bit, then prefix slice per bit
        - progressive heads (progressive=True):
            shared stem: input_dim -> hidden_dim
            per-bit head: hidden_dim -> bit (independent projection per bit)

    Returns list of dicts (one per bit length), each containing:
        continuous: tanh(bn(raw)) in [-1, 1]
        binary: sign(continuous) in {-1, +1}
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        bit_list: list[int] | None = None,
        dropout: float = 0.1,
        progressive: bool = False,
    ):
        super().__init__()
        if bit_list is None:
            bit_list = [8, 16, 32, 48, 64, 128]
        self.bit_list = sorted(bit_list)
        self.max_bit = self.bit_list[-1]
        self.progressive = progressive

        if progressive:
            # Shared stem (no final projection — each head does its own)
            self.stem = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # Per-bit projection heads
            self.heads = nn.ModuleDict(
                {str(bit): nn.Linear(hidden_dim, bit) for bit in self.bit_list}
            )
            for head in self.heads.values():
                nn.init.xavier_uniform_(head.weight, gain=0.5)
                nn.init.zeros_(head.bias)
        else:
            # Original prefix slicing architecture
            self.hash_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.max_bit),
            )
            # Moderate init so tanh doesn't saturate immediately but can reach ±1
            nn.init.xavier_uniform_(self.hash_head[-1].weight, gain=0.5)
            nn.init.zeros_(self.hash_head[-1].bias)

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(bit) for bit in self.bit_list]
        )

    def forward(
        self, embeddings: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        if self.progressive:
            stem_out = self.stem(embeddings)
            outputs = []
            for bit, bn in zip(self.bit_list, self.batch_norms):
                raw = self.heads[str(bit)](stem_out)
                # BatchNorm only — no L2 norm. This lets individual dimensions
                # reach large magnitudes so tanh can saturate toward ±1,
                # producing stable binary codes after sign quantization.
                bn_out = bn(raw)
                continuous = torch.tanh(bn_out)
                binary = SignSTE.apply(continuous)
                outputs.append({"continuous": continuous, "binary": binary})
            return outputs
        else:
            raw = self.hash_head(embeddings)  # (B, max_bit)
            outputs = []
            for length, bn in zip(self.bit_list, self.batch_norms):
                sliced = raw[:, :length]  # prefix slicing
                bn_out = bn(sliced)
                continuous = torch.tanh(bn_out)
                binary = SignSTE.apply(continuous)
                outputs.append({"continuous": continuous, "binary": binary})
            return outputs
