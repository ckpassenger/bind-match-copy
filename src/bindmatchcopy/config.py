"""Model configuration and factory functions.

This module is the single source of truth for how a model is assembled from
its ``layer_types`` list. The hybrid architecture uses the **Gated DeltaNet**
mixer (layer type ``"gated_deltanet"``) in its recurrent/state branch and
``"transformer"`` in its global-attention branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.hybrid_model import HybridModel


# ======================================================================
# Configuration dataclass
# ======================================================================


@dataclass
class ModelConfig:
    """Complete model configuration.

    The paper's controlled models are 8-layer, width-512, 8-head models with
    an FFN of width 2048. The hybrid layout places Gated-DeltaNet layers in the
    bottom/top recurrent stacks and full transformer layers at positions 3 and 7
    (the "global" layers). See ``hybrid_gdn_tf_config`` below.
    """

    # --- architecture ---
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size (used by probes)
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0

    # layer composition: length == n_layers
    # Valid names: "transformer", "gated_deltanet", "sliding_window" (plus
    # window-size aliases "swa_4/8/16/..."). See _LAYER_REGISTRY in hybrid_model.
    layer_types: list[str] = field(default_factory=lambda: ["transformer"] * 8)

    # --- Gated DeltaNet (GDN) ---
    # fla's GatedDeltaNet controls the key dimension through ``head_dim`` rather
    # than ``expand_k``. ``gdn_head_dim=None`` uses fla's default.
    gdn_head_dim: int | None = None
    gdn_expand_v: float = 2.0  # fla default for GDN
    la_conv_kernel_size: int = 4  # short-conv kernel for the GDN mixer

    # --- sliding window ---
    sliding_window_size: int = 4  # window size includes the current token

    # --- RoPE ---
    use_rope: bool = False  # RoPE for transformer / SWA layers
    swa_use_rope: bool = True  # explicit legacy swa_N alias behavior

    # --- output head ---
    output_type: str = "generation"  # "classification" | "generation"
    num_classes: int = 64

    def validate(self):
        from .models.hybrid_model import _LAYER_REGISTRY

        for key in (
            "vocab_size",
            "d_model",
            "n_heads",
            "d_ff",
            "n_layers",
            "max_seq_len",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if len(self.layer_types) != self.n_layers:
            raise ValueError("layer_types must have exactly n_layers entries")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if set(self.layer_types) - _LAYER_REGISTRY.keys():
            raise ValueError(
                f"Unsupported layer types: {set(self.layer_types) - _LAYER_REGISTRY.keys()}"
            )
        if (
            not 0 <= self.dropout < 1
            or self.sliding_window_size < 1
            or self.la_conv_kernel_size < 1
        ):
            raise ValueError("Invalid dropout, window, or convolution width")
        if self.output_type not in ("generation", "classification"):
            raise ValueError("Unknown output_type")
        if (
            self.use_rope
            or (
                self.swa_use_rope
                and any(t.startswith("swa_") for t in self.layer_types)
            )
        ) and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        return self

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
        return cls(**values).validate()


# ======================================================================
# Factory
# ======================================================================


def build_model(config: ModelConfig) -> HybridModel:
    """Instantiate a :class:`HybridModel` from a :class:`ModelConfig`."""
    from .models.hybrid_model import HybridModel

    config.validate()
    return HybridModel(config)


# ======================================================================
# Preset config generators
# ======================================================================


def _apply_overrides(cfg: ModelConfig, overrides: dict[str, Any]) -> ModelConfig:
    """Apply keyword overrides to a config, rejecting unknown keys."""
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg


def pure_tf_config(**overrides) -> ModelConfig:
    """All-Transformer configuration (the pure-transformer control arm)."""
    n = overrides.get("n_layers", 8)
    cfg = ModelConfig(n_layers=n, layer_types=["transformer"] * n)
    return _apply_overrides(cfg, overrides)


def hybrid_gdn_tf_config(**overrides) -> ModelConfig:
    """Hybrid GDN-TF configuration.

    Recurrent (Gated DeltaNet) layers occupy the bottom stack (0-2) and the top
    stack (4-6); full transformer layers occupy the "global" positions 3 and 7.
    This is the paper's canonical 8-layer hybrid layout (GDN = Gated DeltaNet).
    """
    n = overrides.get("n_layers", 8)
    # Default 8-layer layout: GDN bottom (0-2) + TF global (3) + GDN top (4-6) + TF (7)
    layer_types = [
        "gated_deltanet",
        "gated_deltanet",
        "gated_deltanet",
        "transformer",
        "gated_deltanet",
        "gated_deltanet",
        "gated_deltanet",
        "transformer",
    ]
    # If a caller requests a different layer count, fall back to the interleaved
    # GDN/TF pattern (block hybrid) rather than silently truncating the list.
    if n != len(layer_types):
        layer_types = [
            ("gated_deltanet" if i % 4 != 3 else "transformer") for i in range(n)
        ]
    cfg = ModelConfig(n_layers=n, layer_types=layer_types)
    return _apply_overrides(cfg, overrides)


def pure_gdn_config(**overrides) -> ModelConfig:
    """All-Gated-DeltaNet configuration (the pure-recurrent control arm)."""
    n = overrides.get("n_layers", 8)
    cfg = ModelConfig(n_layers=n, layer_types=["gated_deltanet"] * n)
    return _apply_overrides(cfg, overrides)


def swa_tf_config(window_size: int = 4, **overrides) -> ModelConfig:
    """Sliding-window-attention + transformer hybrid.

    The bottom/top recurrent stacks are replaced with ``swa_{window}`` layers,
    keeping the transformer global layers at positions 3 and 7.
    """
    n = overrides.get("n_layers", 8)
    swa = f"swa_{window_size}"
    layer_types = [swa, swa, swa, "transformer", swa, swa, swa, "transformer"]
    if n != len(layer_types):
        layer_types = [(swa if i % 4 != 3 else "transformer") for i in range(n)]
    cfg = ModelConfig(n_layers=n, layer_types=layer_types)
    return _apply_overrides(cfg, overrides)
