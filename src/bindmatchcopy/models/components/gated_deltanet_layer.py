"""Gated DeltaNet wrapper retained from the received server archive.

The recurrence is delegated to the optional flash-linear-attention package.
The archive's parameterization and chunk/padding compatibility behavior are
preserved; exact GPU dependency and checkpoint parity must be established on
the server. This wrapper is not evidence that the paper checkpoints match it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

# fla is an optional dependency: only the GDN mixer (and GPU runs) need it.
# The transformer-only smoke tests run without it.
try:
    from fla.layers import GatedDeltaNet
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Gated DeltaNet mixer requires the 'fla' package. "
        "Install the 'flash-linear-attention' distribution on a supported GPU host."
    ) from exc


class GatedDeltaNetLayer(nn.Module):
    """Pre-norm Gated DeltaNet + FFN, unified layer interface.

    Architecture:
        x -> LayerNorm -> GatedDeltaNet -> residual
          -> LayerNorm -> FFN -> residual

    Numerical compatibility behavior inherited from the archive (GPU-unverified here):
    - fla's GDN triton kernels can produce NaN under pure fp16, so the forward
      pass is wrapped in ``autocast(dtype=bfloat16)``.
    - Short sequences (``seq_len < 64``) are padded to 64 tokens to avoid a
      triton SIGFPE in the chunk kernel (chunk_size = 64).
    - During eval, fla falls back to ``fused_recurrent`` for short sequences,
      whose backward is unimplemented. We force ``train()`` during the GDN call
      and restore the previous mode afterwards so probes run under ``eval()``
      still use the chunk path.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        conv_size: int = 4,
        layer_idx: int = 0,
        expand_v: float = 2.0,
        head_dim: int | None = None,
        use_short_conv: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)

        # fla's GatedDeltaNet controls the key dimension via ``head_dim``
        # (key_dim = num_heads * head_dim), not ``expand_k``.
        if head_dim is None:
            head_dim = max(1, d_model // n_heads)

        self.gdn = GatedDeltaNet(
            mode="chunk",
            hidden_size=d_model,
            num_heads=n_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            use_gate=True,
            allow_neg_eigval=True,
            layer_idx=layer_idx,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ff_up = nn.Linear(d_model, d_ff)
        self.ff_down = nn.Linear(d_ff, d_model)
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
            mask: ignored — Gated DeltaNet is inherently causal.
            collect_aux: reserved for probing (currently returns an empty aux).

        Returns:
            (output, aux).
        """
        h = self.norm1(x)

        # Force chunk-mode (train) so eval-time probes do not hit the
        # unimplemented fused_recurrent backward path.
        device_type = "cuda" if h.is_cuda else "cpu"
        was_training = self.gdn.training
        self.gdn.train()

        # Pad short sequences to the chunk size (64) to avoid triton SIGFPE.
        orig_len = h.shape[1]
        min_len = 64
        if orig_len < min_len:
            pad_len = min_len - orig_len
            h = F.pad(h, (0, 0, 0, pad_len))

        try:
            with autocast(device_type=device_type, dtype=torch.bfloat16):
                gdn_out, _, _ = self.gdn(h)
        finally:
            self.gdn.train(was_training)

        if orig_len < min_len:
            gdn_out = gdn_out[:, :orig_len, :]

        if not was_training:
            self.gdn.eval()
        gdn_out = gdn_out.to(x.dtype)

        x = x + self.resid_drop(gdn_out)

        # FFN.
        h2 = self.norm2(x)
        ff_out = self.ff_down(self.ff_drop(F.gelu(self.ff_up(h2))))
        x = x + self.resid_drop(ff_out)

        return x, {}
