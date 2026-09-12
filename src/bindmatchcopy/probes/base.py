"""Shared primitives for the lag-1 probe family.

The core mechanistic operation is the **position delta patch** (Listing 1 of
the paper): for a given layer and position, capture ``delta = out - in`` (the
block's net residual update, including its FFN) and replace it with a
counterfactual or donor delta, then continue the forward pass.

This is a residual-delta patch rather than a "zero the pre-mixer row" ablation;
it applies uniformly to transformer, Gated DeltaNet, and SWA layers because
they are all residual blocks (``out = x + delta``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from ..interventions.update_patch import UpdatePatch, capture_update


@torch.no_grad()
def logit_margin(
    model, ids: list[int], qpos: int, target: int, distractor: int, device
) -> float:
    """Return ``logit[target] - logit[distractor]`` at position ``qpos``."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x)
    logits = out["logits"][0, qpos, :].float()
    return (logits[target] - logits[distractor]).item()


@torch.no_grad()
def record_pos_delta(model, layer_idx: int, pos: int, ids: list[int], device) -> Tensor:
    """Capture the net residual delta ``(out - in)[pos]`` of ``layer_idx``.

    The hook runs on the layer module; ``inp[0]`` is the layer input and
    ``outp[0]`` is the layer output.
    """
    with capture_update(model.layers[layer_idx], pos) as captured:
        model(torch.tensor([ids], dtype=torch.long, device=device))
    return captured["update"]


class PosDeltaPatch(UpdatePatch):
    def __init__(self, model, layer_idx, pos, cf_delta, rms_match_to=None):
        super().__init__(model.layers[layer_idx], pos, cf_delta, rms_match_to)


@torch.no_grad()
def margin_with_pos_patch(
    model,
    ids: list[int],
    qpos: int,
    target: int,
    distractor: int,
    device,
    layer_idx: int,
    pos: int,
    cf_delta: Tensor,
    rms_match_to: Tensor | None = None,
) -> float:
    """Patch ``(layer_idx, pos)`` delta and return the resulting logit margin."""
    with PosDeltaPatch(model, layer_idx, pos, cf_delta, rms_match_to) as _:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(x)
        logits = out["logits"][0, qpos, :].float()
    return (logits[target] - logits[distractor]).item()


def find_recurrent_module(layer) -> torch.nn.Module:
    """Return the recurrent submodule of a layer (GDN's ``gdn`` attribute)."""
    for attr in ("gdn", "deltanet", "gla"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise AttributeError(f"Cannot find recurrent module in {type(layer).__name__}")


@torch.no_grad()
def query_attn_at(model, ids: list[int], qpos: int, layer_idx: int, device):
    """Return the attention weights at query position ``qpos`` of ``layer_idx``.

    Only transformer / SWA layers expose attention weights (via ``collect_aux``).
    Returns a ``(H, L, L)`` tensor (CPU, float) or ``None`` if unsupported.
    """
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x, collect_aux=True)
    layer_aux = out.get("layer_aux", [])
    if layer_idx >= len(layer_aux):
        return None
    aux = layer_aux[layer_idx]
    if not (isinstance(aux, dict) and aux.get("attn_weights") is not None):
        return None
    return aux["attn_weights"].float().cpu()[0]


def gdn_layer_indices(model) -> list[int]:
    """Return indices of Gated DeltaNet layers in ``model``."""
    types = model.get_layer_types()
    return [i for i, t in enumerate(types) if t.startswith("gated_deltanet")]


def tf_layer_indices(model) -> list[int]:
    """Return indices of transformer (global attention) layers in ``model``."""
    types = model.get_layer_types()
    return [i for i, t in enumerate(types) if t == "transformer"]
