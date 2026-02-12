from __future__ import annotations

import torch
import torch.nn as nn


class EAQLLoss(nn.Module):
    """Equilibrium-Aware Quantization Loss.

    Uses EMA-tracked per-bit importance to weight the quantization error,
    focusing optimization on bits that are hardest to binarize.

    L = mean_over_batch( sum_over_bits( w_d * (h_d - sign(h_d))^2 ) )
    where w_d = ema_d / sum(ema)
    """

    def __init__(self, ema_decay: float = 0.99, bit_list: list[int] | None = None):
        super().__init__()
        self.ema_decay = ema_decay
        # Register EMA buffers so they move with .to(device) and persist in state_dict
        self._ema_keys: set[int] = set()
        for bit in (bit_list or []):
            self._register_ema_buffer(bit)

    def _register_ema_buffer(self, bit: int) -> None:
        """Lazily register an EMA buffer for a given bit length."""
        if bit not in self._ema_keys:
            self.register_buffer(f"_ema_{bit}", None, persistent=True)
            self._ema_keys.add(bit)

    def _get_ema(self, bit: int) -> torch.Tensor | None:
        return getattr(self, f"_ema_{bit}", None)

    def _set_ema(self, bit: int, value: torch.Tensor) -> None:
        if bit not in self._ema_keys:
            self._register_ema_buffer(bit)
        setattr(self, f"_ema_{bit}", value)

    def forward(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous: (B, D) continuous hash codes (e.g. tanh output)

        Returns:
            Scalar EAQL loss.
        """
        with torch.no_grad():
            sign_target = torch.sign(continuous.detach())

        quant_error = (continuous - sign_target) ** 2  # (B, D)
        bitwise_error = quant_error.mean(dim=0)  # (D,)

        bit = continuous.size(1)
        ema = self._get_ema(bit)
        if ema is None:
            self._set_ema(bit, bitwise_error.detach().clone())
        else:
            self._set_ema(
                bit,
                self.ema_decay * ema + (1 - self.ema_decay) * bitwise_error.detach(),
            )

        ema = self._get_ema(bit)
        weights = ema / (ema.sum() + 1e-6)  # normalize to sum=1

        weighted_error = (quant_error * weights).sum(dim=1)  # (B,)
        return weighted_error.mean()
