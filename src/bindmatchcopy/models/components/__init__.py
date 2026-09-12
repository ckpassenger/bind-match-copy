"""Mixer components and shared submodules used by the hybrid model shell."""

from .transformer_layer import TransformerLayer, YaRNConfig
from .sliding_window_layer import SlidingWindowLayer
from .embedding import TokenEmbedding
from .head import TaskHead

__all__ = [
    "TransformerLayer",
    "YaRNConfig",
    "GatedDeltaNetLayer",
    "SlidingWindowLayer",
    "TokenEmbedding",
    "TaskHead",
]


def __getattr__(name):
    if name == "GatedDeltaNetLayer":
        from .gated_deltanet_layer import GatedDeltaNetLayer

        return GatedDeltaNetLayer
    raise AttributeError(name)
