from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hash_layer import SignSTE


class NestedHashLayer(nn.Module):
    """Multi-resolution hash layer with prefix slicing.

    Produces hash codes at multiple bit lengths from a single projection.
    Shorter codes are strict prefixes of longer codes, enabling
    coarse-to-fine retrieval.

    Architecture:
        input_dim -> hidden_dim (LayerNorm + GELU + Dropout) -> max_bit
        Then for each bit length: prefix slice -> BatchNorm -> L2 norm

    Returns list of dicts (one per bit length), each containing:
        continuous: tanh(normalized) in [-1, 1]
        binary: sign(normalized) in {-1, +1}
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        bit_list: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if bit_list is None:
            bit_list = [16, 32, 64, 128]
        self.bit_list = sorted(bit_list)
        self.max_bit = self.bit_list[-1]

        self.hash_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_bit),
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(bit) for bit in self.bit_list]
        )

        # Small weights on final layer for stable output
        nn.init.xavier_uniform_(self.hash_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.hash_head[-1].bias)

    def forward(
        self, embeddings: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        raw = self.hash_head(embeddings)  # (B, max_bit)

        outputs = []
        for length, bn in zip(self.bit_list, self.batch_norms):
            sliced = raw[:, :length]  # prefix slicing
            normalized = F.normalize(bn(sliced), p=2, dim=1)
            continuous = torch.tanh(normalized)
            binary = SignSTE.apply(normalized)
            outputs.append({"continuous": continuous, "binary": binary})

        return outputs
