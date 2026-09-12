"""Released-model adapters (Qwen3 / Qwen3.5 probe integration)."""

from .qwen import (
    load_qwen_model,
    prepare_tokens,
    make_pair,
    layer_kind,
    logit_margin,
    record_pos_delta,
    PosDeltaPatch,
    margin_with_pos_patch,
    query_attn_at,
    edge_mask_patch,
    value_swap_patch,
    capture_value_at,
    QUERY_POS,
)

__all__ = [
    "load_qwen_model",
    "prepare_tokens",
    "make_pair",
    "layer_kind",
    "logit_margin",
    "record_pos_delta",
    "PosDeltaPatch",
    "margin_with_pos_patch",
    "query_attn_at",
    "edge_mask_patch",
    "value_swap_patch",
    "capture_value_at",
    "QUERY_POS",
]
