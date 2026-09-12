"""Standard Transformer layer with pre-norm causal self-attention + FFN.

This is the "global" mixer of the hybrid architecture. It also supports:

- Rotary Position Embedding (RoPE) with an optional YaRN length-extrapolation
  extension.
- A local sliding-window mode (``window_size``) used by the SWA control arms.
- Head-level interventions (``disabled_heads`` / ``head_scale`` /
  ``head_lag_limit``) used by the causal route and formation experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .attention import effective_mask, attend


@dataclass
class YaRNConfig:
    """YaRN length-extrapolation configuration (optional; disabled by default)."""

    enabled: bool = False
    original_max_seq_len: int = 256
    target_max_seq_len: int = 512
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attn_factor: float | None = None  # None = auto-compute

    @property
    def scale(self) -> float:
        return self.target_max_seq_len / self.original_max_seq_len

    def get_attn_factor(self) -> float:
        if self.attn_factor is not None:
            return self.attn_factor
        return 0.1 * math.log(self.scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float = 10000.0
) -> float:
    return (dim * math.log(num_rotations / (2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    original_max_seq_len: int,
) -> tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(
            beta_fast * original_max_seq_len / (2 * math.pi), dim, base
        )
    )
    high = math.ceil(
        _yarn_find_correction_dim(
            beta_slow * original_max_seq_len / (2 * math.pi), dim, base
        )
    )
    low = max(low, 0)
    high = min(max(high, 0), dim // 2 - 1)
    if low > high:
        low = high
    return low, high


def _yarn_linear_ramp_mask(
    low: int, high: int, dim_half: int, dtype: torch.dtype
) -> Tensor:
    if low == high:
        high = low + 1
    ramp = torch.zeros(dim_half, dtype=dtype)
    indices = torch.arange(dim_half, dtype=dtype)
    ramp = ((indices - low).clamp(0) / (high - low)).clamp(0, 1)
    return ramp


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE), with optional YaRN extension."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0,
        yarn_config: YaRNConfig | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.yarn_config = yarn_config

        if yarn_config is not None and yarn_config.enabled:
            inv_freq = self._build_yarn_inv_freq(dim, base, yarn_config)
            self._attn_factor = yarn_config.get_attn_factor()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self._attn_factor = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    @staticmethod
    def _build_yarn_inv_freq(dim: int, base: float, cfg: YaRNConfig) -> Tensor:
        s = cfg.scale
        dim_half = dim // 2
        inv_freq_orig = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq_interp = inv_freq_orig / s
        low, high = _yarn_find_correction_range(
            cfg.beta_fast, cfg.beta_slow, dim, base, cfg.original_max_seq_len
        )
        ramp = _yarn_linear_ramp_mask(low, high, dim_half, inv_freq_orig.dtype)
        return (1 - ramp) * inv_freq_orig + ramp * inv_freq_interp

    def _update_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
            self._sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

    @property
    def attn_factor(self) -> float:
        return self._attn_factor

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        L = q.size(2)
        self._update_cache(L, q.device)
        cos = self._cos_cached[:, :, :L, :].to(q.dtype)
        sin = self._sin_cached[:, :, :L, :].to(q.dtype)
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TransformerLayer(nn.Module):
    """Pre-norm Transformer block (global mixer).

    Structure: x -> LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual.
    """

    supports_attention_interventions = True

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        use_rope: bool = False,
        yarn_config: YaRNConfig | None = None,
        window_size: int | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rope = use_rope
        self.window_size = window_size

        # Head-level interventions (used by the causal-route / formation probes):
        #   head_lag_limit: dict[int, int]  head_idx -> exact lag (plus self)
        #   disabled_heads: set[int]        heads whose attention output is zeroed
        #   head_scale:     dict[int, float] head_idx -> output scale (0..1)
        self.head_lag_limit = None
        self.disabled_heads = None
        self.head_scale = None
        self._yarn_attn_factor = 1.0

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

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        collect_aux: bool = False,
    ) -> tuple[Tensor, dict]:
        """Forward pass.

        Args:
            x: (B, L, D).
            mask: (1, 1, L, L) causal mask (True = keep). Intersected with any window
                or head-lag mask is active.
            collect_aux: return attention weights for probing.

        Returns:
            (output, aux) where ``aux["attn_weights"]`` is (B, H, L, L) if
            requested.
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

        # FFN.
        h = self.norm2(x)
        ff_out = self.ff_down(self.ff_drop(F.gelu(self.ff_up(h))))
        x = x + self.resid_drop(ff_out)

        aux: dict = {}
        if collect_aux:
            aux["attn_weights"] = attn_weights.detach()
        return x, aux
