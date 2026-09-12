"""One attention computation for native forward and mechanistic interventions."""

import torch
import torch.nn.functional as F


def effective_mask(layer, length, device, mask=None):
    rows = torch.arange(length, device=device)[:, None]
    cols = torch.arange(length, device=device)[None, :]
    distance = rows - cols
    keep = (distance >= 0)[None, None]
    window = getattr(layer, "window_size", None)
    if window is not None:
        keep = keep & (distance < window)[None, None]
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("Attention mask must be boolean (True = visible)")
        keep = keep & mask
    limits = getattr(layer, "head_lag_limit", None)
    if limits:
        keep = keep.expand(keep.shape[0], layer.n_heads, length, length).clone()
        for head, lag in limits.items():
            if not 0 <= head < layer.n_heads or lag < 1:
                raise ValueError("Invalid local-head coordinate or lag")
            # Wrong-lag=2 means self and lag TWO, excluding lag one.
            keep[:, head] &= (distance == 0) | (distance == lag)
    if not keep.any(dim=-1).all():
        raise ValueError("Attention mask contains an empty query row")
    return keep


def attend(layer, q, k, v, mask, scale, collect_aux):
    patch = getattr(layer, "_bmc_attention_patch", None)
    if patch is not None:
        q, k, v = patch.qkv(q, k, v)
    if collect_aux or patch is not None:
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        p = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        if patch is not None:
            p = patch.pattern(p)
        p = layer.attn_drop(p)
        output = (p @ v.float()).to(v.dtype)
    else:
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            scale=scale,
            dropout_p=layer.attn_drop.p if layer.training else 0.0,
        )
        p = None
    if patch is not None:
        output = patch.output(output)
    disabled = getattr(layer, "disabled_heads", None) or set()
    scales = getattr(layer, "head_scale", None) or {}
    for head in set(disabled) | set(scales):
        if not 0 <= head < layer.n_heads:
            raise ValueError("Invalid attention head")
    gain = output.new_ones(layer.n_heads)
    for head, value in scales.items():
        gain[head] = value
    for head in disabled:
        gain[head] = 0.0
    return output * gain[None, :, None, None], p
