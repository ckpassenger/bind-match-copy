"""Per-layer-type optimizer parameter groups (the backbone of LR damp).

The formation-time "damp" interventions scale the learning rate of the
recurrent (GDN) branch independently of the transformer branch. To do that the
optimizer needs one parameter group per branch. This module classifies each
parameter as ``gdn``, ``tf``, or ``other`` based on its layer type, and further
splits into decayed / non-decayed groups so weight decay is not applied to
biases and norms.
"""

from __future__ import annotations

import torch.nn as nn


# Layer types that belong to the recurrent / state branch (GDN family).
GDN_LAYER_TYPES = {
    "deltanet",
    "deltanet_noconv",
    "deltanet_conv2",
    "deltanet_conv3",
    "deltanet_skip_lag1",
    "gdn",
    "gla",
    "gated_deltanet",
    "gated_deltanet_noconv",
    "la_conv",
}

# Layer types that belong to the attention branch (transformer / SWA family).
TF_LAYER_TYPES = {
    "transformer",
    "tf",
    "sliding_window",
    "swa_2",
    "swa_4",
    "swa_8",
    "swa_16",
    "swa_32",
    "swa_64",
    "tf_win32",
    "tf_win64",
    "tf_win96",
    "tf_win128",
    "dsa_tf",
}


def _is_non_decayed(name: str, param: nn.Parameter) -> bool:
    """Return True for biases and norm params (no weight decay)."""
    return param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()


def build_param_groups(
    model: nn.Module, layer_types: list[str], weight_decay: float = 0.1
):
    """Build optimizer param groups keyed by ``group_name`` in {gdn, tf, other}.

    Each parameter of a layer is assigned to a branch according to that layer's
    type (see ``GDN_LAYER_TYPES`` / ``TF_LAYER_TYPES``). Non-decayed parameters
    (biases / norms) get ``weight_decay=0``.

    Returns:
        A list of dicts ``{"params": [...], "weight_decay": float, "group_name": str}``
        suitable for ``torch.optim.AdamW``.
    """
    groups = {}

    # Map parameter-name prefixes (``layers.<i>``) to a branch.
    type_map: dict[str, str] = {}
    for i, lt in enumerate(layer_types):
        prefix = f"layers.{i}."  # L1 must not match L10/L11/L12...
        if lt in GDN_LAYER_TYPES:
            type_map[prefix] = "gdn"
        elif lt.startswith("swa_") or lt == "sliding_window":
            type_map[prefix] = "swa"
        elif lt in TF_LAYER_TYPES:
            type_map[prefix] = "tf"

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        branch = "other"
        layer_index = None
        for prefix, bt in type_map.items():
            if name.startswith(prefix):
                branch = bt
                layer_index = int(prefix.split(".")[1])
                break
        nd = _is_non_decayed(name, param)
        groups.setdefault((branch, layer_index, nd), []).append((name, param))

    param_groups = []
    for (branch, layer_index, nd), named in groups.items():
        params = [p for _, p in named]
        if not params:
            continue
        wd = 0.0 if nd else weight_decay
        param_groups.append(
            {
                "params": params,
                "weight_decay": wd,
                "group_name": branch,
                "layer_index": layer_index,
                "parameter_names": [name for name, _ in named],
            }
        )
    return param_groups
