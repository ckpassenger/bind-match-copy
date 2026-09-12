"""Qwen3 / Qwen3.5 adapter for the lag-1 probe family.

The released-model experiments (suite ``07_released_qwen``) reuse the same lag-1
probe primitives as the controlled models, but operate on HuggingFace Qwen
checkpoints instead of the project's ``HybridModel``. This adapter wraps the
transformers Qwen3 / Qwen3.5 modules and exposes the same operations:

- ``logit_margin`` — next-token logit margin at the query position.
- ``record_pos_delta`` / ``PosDeltaPatch`` / ``margin_with_pos_patch`` — the
  residual-delta patch primitive (works on both full-attention and linear layers
  because both are residual blocks).
- ``query_attn_at`` — attention weights at the query (full-attention layers only).
- ``edge_mask_patch`` / ``value_swap_patch`` / ``capture_value_at`` — attention
  K-edge and V-copy interventions (full-attention layers only).

Layer-kind helper :func:`layer_kind` distinguishes Qwen3.5's ``linear_attention``
(recurrent / Gated-DeltaNet style) layers from its ``full_attention`` layers so
that attention-specific probes can record ``unsupported`` instead of fabricating
a zero result (as required by the architecture contract).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...data.prompts import (
    make_pair as _base_make_pair,
)  # noqa: F401  (kept for parity)

# Import transformers lazily so the transformer-only path has no hard dependency.
try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    _HAS_QWEN = True
except Exception:  # pragma: no cover
    _HAS_QWEN = False

# Default released-model path (overridden per call).
DEFAULT_MODEL_PATH = None  # caller supplies a pinned local checkpoint or model ID

# Needle / filler token construction for released models (single-token needles).
DEFAULT_NEEDLE_WORDS = [
    "apple",
    "banana",
    "castle",
    "rocket",
    "fish",
    "bird",
    "tree",
    "forest",
    "bridge",
    "window",
]
DEFAULT_FILLER_TEXT = (
    "the quick brown fox jumps over the lazy dog and then we see a cat "
    "running fast away from home while people walk slowly across the street "
    "in the morning light of a quiet town near the river"
)

# Query position for the released-model probe sequence (sequence length 512).
QUERY_POS = 511


def _require_qwen() -> None:
    if not _HAS_QWEN:
        raise ImportError(
            "The Qwen adapter requires the 'transformers' package and the Qwen3 "
            "modeling module. Install with `pip install transformers`."
        )


def load_qwen_model(model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda:0"):
    """Load a Qwen3 / Qwen3.5 causal LM and its tokenizer.

    Returns ``(model, tokenizer, config)``. The model is loaded in bfloat16 with
    eager attention (required so attention weights can be read by probes).
    """
    if not model_path:
        raise ValueError("An explicit Qwen model path or ID is required")
    _require_qwen()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config


def prepare_tokens(tokenizer, needle_words=None, filler_text=None):
    """Pick four single-token needles and a filler token list.

    Returns ``(needle, filler)``. Needles are the first four words in
    ``needle_words`` that tokenize to a single (unique) token id.
    """
    words = needle_words or DEFAULT_NEEDLE_WORDS
    text = filler_text or DEFAULT_FILLER_TEXT
    needle = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in needle:
            needle.append(ids[0])
        if len(needle) == 4:
            break
    if len(needle) < 4:
        raise RuntimeError(f"only {len(needle)} single-token needles found")
    filler = list(dict.fromkeys(tokenizer.encode(text, add_special_tokens=False)))
    return needle, filler


def make_pair(
    kind: str, rng, dist: int, needle: list[int], filler: list[int], base_ids=None
) -> dict:
    """Build a released-model probe sequence (mirrors ``data.prompts.make_pair``).

    Uses ``QUERY_POS = 511`` and a 512-token sequence.
    """
    A, B, C, D = needle
    q = QUERY_POS
    r2 = q - dist
    r1 = max(2, r2 - max(8, dist // 2))
    ids = (
        list(base_ids)
        if base_ids is not None
        else rng.choice(filler, size=512, replace=True).tolist()
    )
    A2 = None
    if kind == "clean":
        ids[r1] = A
        ids[r1 + 1] = B
        ids[r2] = C
        ids[r2 + 1] = D
        ids[q] = A
        T, Dst = B, D
    elif kind == "counterfactual_A":
        A2 = A
        while A2 == A:
            A2 = int(rng.choice(filler))
        ids[r1] = A2
        ids[r1 + 1] = B
        ids[r2] = C
        ids[r2 + 1] = D
        ids[q] = A
        T, Dst = B, D
    elif kind == "key_swap":
        ids[r1] = C
        ids[r1 + 1] = B
        ids[r2] = A
        ids[r2 + 1] = D
        ids[q] = A
        T, Dst = B, D
    elif kind == "position_swap":
        ids[r1] = C
        ids[r1 + 1] = D
        ids[r2] = A
        ids[r2 + 1] = B
        ids[q] = A
        T, Dst = B, D
    else:
        raise ValueError(f"Unknown pair kind: {kind}")
    out = {
        "ids": ids,
        "r1": r1,
        "r2": r2,
        "q": q,
        "A1_pos": r1,
        "B1_pos": r1 + 1,
        "C1_pos": r2,
        "D1_pos": r2 + 1,
        "Aq_pos": q,
        "T": T,
        "Dst": Dst,
    }
    if A2 is not None:
        out["A2"] = A2
    return out


def layer_kind(model, layer_idx: int) -> str:
    """Return "full_attention" or "linear_attention" for a Qwen3.5 layer.

    Qwen3 (pure transformer) layers are always "full_attention". Qwen3.5 layers
    expose ``self_attn`` (full) or ``linear_attn`` (recurrent) submodules.
    """
    layer = model.model.layers[layer_idx]
    if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
        return "linear_attention"
    return "full_attention"


@torch.no_grad()
def logit_margin(model, ids: list[int], qpos: int, T: int, D: int, device) -> float:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x, use_cache=False)
    logits = out.logits[0, qpos, :].float()
    return (logits[T] - logits[D]).item()


@torch.no_grad()
def record_pos_delta(
    model, layer_idx: int, pos: int, ids: list[int], device
) -> torch.Tensor:
    """Capture the residual delta ``(out - in)[pos]`` of a Qwen transformer layer."""
    from ...interventions.update_patch import capture_update

    with capture_update(model.model.layers[layer_idx], pos) as captured:
        model(torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
    return captured["update"]


from ...interventions.update_patch import UpdatePatch


class PosDeltaPatch(UpdatePatch):
    def __init__(self, model, layer_idx, pos, cf_delta, rms_match_to=None):
        super().__init__(model.model.layers[layer_idx], pos, cf_delta, rms_match_to)


@torch.no_grad()
def margin_with_pos_patch(
    model, ids, qpos, T, D, device, layer_idx, pos, cf_delta, rms_match_to=None
) -> float:
    with PosDeltaPatch(model, layer_idx, pos, cf_delta, rms_match_to):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(x, use_cache=False)
        logits = out.logits[0, qpos, :].float()
    return (logits[T] - logits[D]).item()


@torch.no_grad()
def query_attn_at(model, ids, qpos, layer_idx, device):
    """Return attention weights at the query (full-attention layers only)."""
    out = model(
        torch.tensor([ids], dtype=torch.long, device=device),
        output_attentions=True,
        use_cache=False,
    )
    attn = out.attentions[layer_idx]
    return attn.float().cpu()[0]


def _attn_pre(attn, hidden_states, position_embeddings):
    """Project and RoPE-rotate Q/K/V for a Qwen attention module (GQA)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, input_shape


def _causal_mask(L: int, device, dtype):
    m = torch.full((L, L), float("-inf"), device=device, dtype=torch.float32)
    return torch.triu(m, diagonal=1).to(dtype)


def edge_mask_patch(attn, src_pos: int, qpos: int):
    """Return ``(patched_forward, orig_forward)`` masking the query->src edge.

    Full-attention layers only (a linear layer has no K edge to mask).
    """
    _require_qwen()
    orig = attn.forward

    def patched(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        q, k, v, input_shape = _attn_pre(attn, hidden_states, position_embeddings)
        k_rep = repeat_kv(k, attn.num_key_value_groups)
        v_rep = repeat_kv(v, attn.num_key_value_groups)
        L = q.shape[2]
        scores = torch.matmul(q, k_rep.transpose(2, 3)) * attn.scaling
        scores = scores + _causal_mask(L, q.device, q.dtype)[None, None, :, :]
        scores[:, :, qpos, src_pos] = float("-inf")
        aw = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(aw, v_rep).transpose(1, 2).contiguous()
        out = out.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(out), None

    return patched, orig


def value_swap_patch(attn, src_pos: int, control_v, head_idx=None):
    """Return ``(patched_forward, orig_forward)`` swapping V at ``src_pos``.

    Full-attention layers only.
    """
    _require_qwen()
    orig = attn.forward

    def patched(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        q, k, v, input_shape = _attn_pre(attn, hidden_states, position_embeddings)
        v = v.clone()
        if head_idx is None:
            v[:, :, src_pos, :] = control_v.to(v.device, v.dtype)
        else:
            v[:, head_idx, src_pos, :] = control_v.to(v.device, v.dtype)
        k_rep = repeat_kv(k, attn.num_key_value_groups)
        v_rep = repeat_kv(v, attn.num_key_value_groups)
        L = q.shape[2]
        scores = torch.matmul(q, k_rep.transpose(2, 3)) * attn.scaling
        scores = scores + _causal_mask(L, q.device, q.dtype)[None, None, :, :]
        aw = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(aw, v_rep).transpose(1, 2).contiguous()
        out = out.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(out), None

    return patched, orig


@torch.no_grad()
def capture_value_at(model, layer_idx: int, ids, pos: int, device):
    """Capture the value (V) at ``pos`` of a Qwen full-attention layer."""
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inp, outp):
        x = inp[0] if isinstance(inp, tuple) else inp
        hn = layer.input_layernorm(x)
        input_shape = hn.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        v = attn.v_proj(hn).view(hidden_shape).transpose(1, 2)
        captured["V"] = v[0, :, pos, :].detach().clone()

    h = layer.register_forward_hook(hook)
    try:
        model(torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
    finally:
        h.remove()
    return captured["V"]


# ═══════════════════════════════════════════════════════════════════
# Qwen3.5 (hybrid) component-decomposition primitives
# ═══════════════════════════════════════════════════════════════════
#
# Qwen3.5-4B is a hybrid model (``Qwen3_5ForConditionalGeneration``) whose text
# trunk interleaves ``linear_attention`` (Gated-DeltaNet-style) layers and
# ``full_attention`` layers (GQA with an extra *gate* channel). These helpers
# expose the Q/K/V/gate projections and the gated attention forward so the
# component-decomposition probe (``probes.components``) can isolate which
# channel carries the Bind -> Match/Use information.

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        apply_rotary_pos_emb as _qwen35_rope,
        repeat_kv as _qwen35_repeat_kv,
    )

    _HAS_QWEN35 = True
except Exception:  # pragma: no cover
    _HAS_QWEN35 = False


def _require_qwen35() -> None:
    if not _HAS_QWEN35:
        raise ImportError(
            "The Qwen3.5 component-decomposition probe requires the "
            "'transformers' package with the Qwen3.5 modeling module."
        )


def load_qwen3_5_model(model_path: str, device: str = "cuda:0"):
    """Load a Qwen3.5 hybrid causal LM (``Qwen3_5ForConditionalGeneration``).

    Returns ``(model, tokenizer, config)``. The model is loaded in bfloat16 with
    eager attention (required so the Q/K/V projections can be read by probes).
    """
    _require_qwen35()
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config


def qwen35_text_layer(model, layer_idx: int):
    """Return the ``layer_idx``-th *text* layer of a Qwen3.5 model."""
    return model.model.language_model.layers[layer_idx]


def qwen35_layer_kind(model, layer_idx: int) -> str:
    """Return ``"linear_attention"`` or ``"full_attention"`` for a Qwen3.5 layer."""
    return getattr(qwen35_text_layer(model, layer_idx), "layer_type", None)


def qwen35_attn_pre(attn, hidden_states, position_embeddings):
    """Project Q/K/V (with q_norm/k_norm/rope) and split the gate channel.

    Qwen3.5 full-attention projects ``q_proj`` to ``2 * head_dim``; the second
    half is the per-token gate. Returns ``(q, k, v, input_shape, gate)``.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query_states, gate = torch.chunk(
        attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = _qwen35_rope(query_states, key_states, cos, sin)
    return query_states, key_states, value_states, input_shape, gate


def qwen35_normal_forward(attn, q, k, v, input_shape, gate):
    """Run the gated GQA attention forward from projected Q/K/V/gate."""
    k_rep = _qwen35_repeat_kv(k, attn.num_key_value_groups)
    v_rep = _qwen35_repeat_kv(v, attn.num_key_value_groups)
    Ln = q.shape[2]
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * attn.scaling
    causal = torch.triu(
        torch.full((Ln, Ln), float("-inf"), device=scores.device, dtype=torch.float32),
        diagonal=1,
    )
    scores = scores + causal[None, None, :, :].to(scores.dtype)
    aw = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(aw, v_rep).transpose(1, 2).contiguous()
    out = out.reshape(*input_shape, -1).contiguous()
    out = out * torch.sigmoid(gate)
    return attn.o_proj(out), None
