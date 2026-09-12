"""Unified sequence model shell that assembles heterogeneous layers by config.

The shell represents Pure-TF, Pure-GDN, and Hybrid (GDN-TF / SWA-TF)
architectures by simply varying ``config.layer_types``. Every layer shares the
same ``forward(x, mask, collect_aux) -> (Tensor, dict)`` interface so probes
can iterate over layers uniformly.

The hybrid architecture's recurrent branch uses **Gated DeltaNet**
(layer type ``"gated_deltanet"``); its global branch uses ``"transformer"``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .components import (
    TransformerLayer,
    SlidingWindowLayer,
    TokenEmbedding,
    TaskHead,
    YaRNConfig,
)


def _gdn_layer(**kwargs):
    from .components.gated_deltanet_layer import GatedDeltaNetLayer

    return GatedDeltaNetLayer(**kwargs)


# Registry mapping layer-type strings to layer classes or callables.
# New scientific mixers are added here (one line each), keeping experiments
# as configuration rather than code.
_LAYER_REGISTRY: dict[str, type[nn.Module] | callable] = {
    "transformer": TransformerLayer,
    "gated_deltanet": _gdn_layer,
    "gated_deltanet_noconv": lambda **kw: _gdn_layer(use_short_conv=False, **kw),
    "sliding_window": SlidingWindowLayer,
    # Fixed-window SWA aliases (window size includes the current token).
    "swa_2": lambda **kw: SlidingWindowLayer(window_size=2, **kw),
    "swa_4": lambda **kw: SlidingWindowLayer(window_size=4, **kw),
    "swa_8": lambda **kw: SlidingWindowLayer(window_size=8, **kw),
    "swa_16": lambda **kw: SlidingWindowLayer(window_size=16, **kw),
    "swa_32": lambda **kw: SlidingWindowLayer(window_size=32, **kw),
    "swa_64": lambda **kw: SlidingWindowLayer(window_size=64, **kw),
}


class HybridModel(nn.Module):
    """Heterogeneous sequence model assembled from ``config.layer_types``."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Build each layer according to its type.
        self.layer_types: list[str] = list(config.layer_types)
        self.layers = nn.ModuleList()
        use_rope = getattr(config, "use_rope", False)

        # Optional YaRN length-extrapolation config.
        yarn_config = None
        if getattr(config, "yarn_enabled", False):
            yarn_config = YaRNConfig(
                enabled=True,
                original_max_seq_len=getattr(config, "yarn_original_max_seq_len", 256),
                target_max_seq_len=getattr(config, "yarn_target_max_seq_len", 512),
                beta_fast=getattr(config, "yarn_beta_fast", 32.0),
                beta_slow=getattr(config, "yarn_beta_slow", 1.0),
                attn_factor=getattr(config, "yarn_attn_factor", None),
            )

        for i, ltype in enumerate(self.layer_types):
            cls_or_fn = _LAYER_REGISTRY[ltype]
            kwargs = dict(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )

            if ltype == "transformer":
                kwargs["use_rope"] = use_rope
                if yarn_config is not None:
                    kwargs["yarn_config"] = yarn_config
            elif ltype == "gated_deltanet" or ltype == "gated_deltanet_noconv":
                # Gated DeltaNet: layer_idx + short-conv kernel size.
                kwargs["layer_idx"] = i
                kwargs["conv_size"] = config.la_conv_kernel_size
                gdn_head_dim = getattr(config, "gdn_head_dim", None)
                if gdn_head_dim is not None:
                    kwargs["head_dim"] = gdn_head_dim
                gdn_expand_v = getattr(config, "gdn_expand_v", None)
                if gdn_expand_v is not None:
                    kwargs["expand_v"] = gdn_expand_v
            elif ltype == "sliding_window":
                kwargs["window_size"] = config.sliding_window_size
                kwargs["use_rope"] = use_rope
                if yarn_config is not None:
                    kwargs["yarn_config"] = yarn_config
            elif ltype.startswith("swa_"):
                # All SWA layers use RoPE unconditionally (their window is fixed
                # by the registry lambda; the global use_rope flag only governs
                # transformer layers).
                kwargs["use_rope"] = config.swa_use_rope
                if yarn_config is not None:
                    kwargs["yarn_config"] = yarn_config

            self.layers.append(cls_or_fn(**kwargs))

        # Output head.
        head_kwargs = dict(
            d_model=config.d_model,
            output_type=config.output_type,
            num_classes=config.num_classes,
        )
        if config.output_type == "generation":
            head_kwargs["vocab_size"] = config.vocab_size
        self.head = TaskHead(**head_kwargs)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        collect_aux: bool = False,
    ) -> dict:
        """Run the full model.

        Args:
            input_ids: (B, L).
            labels: (B,) for classification / (B, L) for generation / None.
            collect_aux: collect per-layer analysis info.

        Returns:
            dict with "logits", optional "loss", "layer_aux" (list or None),
            and "hidden_states" (list or None).
        """
        x = self.embedding(input_ids)

        B, L, D = x.shape
        causal_mask = (
            torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, L, L)

        layer_aux: list[dict] = [] if collect_aux else None
        hidden_states: list[Tensor] = [] if collect_aux else None

        for layer in self.layers:
            x, aux = layer(x, mask=causal_mask, collect_aux=collect_aux)
            if collect_aux:
                layer_aux.append(aux)
                hidden_states.append(x)

        head_out = self.head(x, labels, input_ids=input_ids)

        result = {"logits": head_out["logits"]}
        if "loss" in head_out:
            result["loss"] = head_out["loss"]
        result["layer_aux"] = layer_aux
        result["hidden_states"] = hidden_states
        return result

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_layer_types(self) -> list[str]:
        """Return the layer-type string of every layer."""
        return list(self.layer_types)

    def get_layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def count_parameters(self) -> dict:
        """Count parameters by layer type plus embedding/head/total."""
        counts: dict[str, int] = {}
        for i, ltype in enumerate(self.layer_types):
            n = sum(p.numel() for p in self.layers[i].parameters())
            counts[ltype] = counts.get(ltype, 0) + n
        counts["embedding"] = sum(p.numel() for p in self.embedding.parameters())
        counts["head"] = sum(p.numel() for p in self.head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
