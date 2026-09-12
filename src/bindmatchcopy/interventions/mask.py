"""Exact-lag local heads: self plus lag-k, not the whole k-token window.

Model checkpoint metadata stores active masks separately from learned weights.
"""

import copy


def _targets(model, layers, head, lag):
    if not layers or len(set(layers)) != len(layers) or lag < 1 or head < 0:
        raise ValueError("Invalid local-head specification")
    result = []
    for i in layers:
        if not 0 <= i < len(model.layers):
            raise ValueError("Local-head layer out of range")
        layer = model.layers[i]
        if not hasattr(layer, "head_lag_limit") or head >= layer.n_heads:
            raise ValueError("Local-head target is not a supported attention head")
        result.append(layer)
    return result


def apply_lag_mask(model, layer_indices, head_idx, lag):
    for layer in _targets(model, layer_indices, head_idx, lag):
        layer.head_lag_limit = {**(layer.head_lag_limit or {}), head_idx: lag}


class LagHeadScaffoldController:
    """Compatibility name for a local-head intervention, inclusive release step."""

    def __init__(self, config, total_steps):
        unknown = set(config) - {"lag", "head_idx", "release_step", "always", "layers"}
        if unknown:
            raise ValueError(f"Unknown local-head fields: {unknown}")
        self.lag = config.get("lag", 1)
        self.head_idx = config.get("head_idx", 0)
        self.release_step = config.get("release_step", 3000)
        self.always = config.get("always", False)
        self.layer_indices = list(config.get("layers", []))
        self.total_steps = total_steps
        self._hooks = []
        self._original = []
        if self.release_step < 1:
            raise ValueError("release_step must be positive")

    def patch_model(self, model, step_holder):
        if self._hooks:
            raise RuntimeError("Local-head controller already installed")
        for layer in _targets(model, self.layer_indices, self.head_idx, self.lag):
            original = copy.deepcopy(layer.head_lag_limit)
            self._original.append((layer, original))

            def hook(module, args, original=original):
                module.head_lag_limit = (
                    {**(original or {}), self.head_idx: self.lag}
                    if self.always or step_holder["step"] <= self.release_step
                    else original
                )

            self._hooks.append(layer.register_forward_pre_hook(hook))
        return model

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        for layer, original in self._original:
            layer.head_lag_limit = original
        self._hooks = []
        self._original = []
