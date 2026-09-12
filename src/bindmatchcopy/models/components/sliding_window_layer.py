"""Sliding Window Attention (SWA) layer.

SWA is the "view-range bottleneck" control mixer: standard multi-head
self-attention whose causal mask is further restricted to a local window of
``window_size`` tokens (including the current token).

In the paper's SWA control arms, the bottom/top recurrent stacks are replaced
with SWA layers while the transformer global layers remain unchanged. SWA keeps
the same ``forward(x, mask, collect_aux)`` signature as the other mixers so it
can be mixed into the same shell.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .attention import effective_mask, attend

from .transformer_layer import RotaryPositionEmbedding, YaRNConfig


class SlidingWindowLayer(nn.Module):
    """Pre-norm multi-head self-attention with a causal sliding window."""

    supports_attention_interventions = True

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        window_size: int = 4,
        dropout: float = 0.0,
        use_rope: bool = False,
        yarn_config: YaRNConfig | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.use_rope = use_rope
        self._yarn_attn_factor = 1.0
        self.head_lag_limit = None
        self.disabled_heads = None
        self.head_scale = None

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        if use_rope:
            self.rope = RotaryPositionEmbedding(self.d_k, yarn_config=yarn_config)
            self._yarn_attn_factor = self.rope.attn_factor

        self.ff_up = nn.Linear(d_model, d_ff)
        self.ff_down = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _build_window_mask(self, L: int, device: torch.device) -> Tensor:
        """Build a causal sliding-window mask (1, 1, L, L); True = keep."""
        rows = torch.arange(L, device=device).unsqueeze(1)
        cols = torch.arange(L, device=device).unsqueeze(0)
        mask = (cols <= rows) & ((rows - cols) < self.window_size)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        collect_aux: bool = False,
    ) -> tuple[Tensor, dict]:
        """Forward pass.

        Args:
            x: (B, L, D).
            mask: boolean visibility mask, intersected with causality and the local window.
            collect_aux: return attention weights.

        Returns:
            (output, aux) where ``aux["attn_weights"]`` is (B, H, L, L).
        """
        h = self.norm1(x)
        B, L, _ = h.shape

        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q, K = self.rope(Q, K)

        eff_mask = effective_mask(self, L, x.device, mask)
        scale = 1.0 / (math.sqrt(self.d_k) * self._yarn_attn_factor)
        attn_out, attn_weights = attend(self, Q, K, V, eff_mask, scale, collect_aux)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.o_proj(attn_out)
        x = x + self.resid_drop(attn_out)

        h2 = self.norm2(x)
        ff_out = self.ff_down(self.ff_drop(F.gelu(self.ff_up(h2))))
        x = x + self.resid_drop(ff_out)

        aux: dict = {}
        if collect_aux:
            aux["attn_weights"] = attn_weights.detach()
        return x, aux
