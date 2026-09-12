"""Formation-time damp / gating / ablation interventions.

Three related but distinct interventions share the same "formation window"
idea — act only during ``[0, release_step]``, then release:

1. :class:`LRDampSchedule` — **soft** learning-rate damp. The GDN branch's LR
   is multiplied by ``attenuation_scale`` (e.g. 0.1) during the window, then
   restored. The optimizer moments are preserved (the LR is changed, not the
   gradient, and the optimizer is not re-created). This is the C2 / R3 / R4
   damp-release family.
2. :class:`FormationGateController` — **forward output gain**. The GDN branch's
   residual output is scaled by a gain during the window (N17 asymmetric
   formation gating). This is an output-scale gate, not an LR damp, and does
   not permanently remove any path.
3. :class:`Lag1AblationController` — **hard** short-conv lag ablation. The GDN
   short-conv weight for the chosen lag is forced to zero during the window
   (N21 lag-1/2 ablation), the hard counterpart of the soft damp.

All three keep their release boundary and per-step semantics explicit so the
checkpoint-evaluation and resume paths can replay them faithfully.
"""

from __future__ import annotations

import torch


class LRDampSchedule:
    """Soft LR-damp schedule for the recurrent (GDN) branch.

    Attributes:
        attenuation_scale: GDN LR multiplier during the window (default 0.1).
        release_step: last step of the window (None = never release).
        release_type: "sharp" (step function) or "ramp" (linear interpolation
            between ``ramp_start`` and ``ramp_end``).
        ramp_start / ramp_end / ramp_start_scale / ramp_end_scale: ramp bounds.
        tf_boost: multiplier applied to the transformer branch LR (default 1.0).
    """

    def __init__(
        self,
        attenuation_scale: float = 0.1,
        release_step: int | None = 3000,
        release_type: str = "sharp",
        ramp_start: int | None = None,
        ramp_end: int | None = None,
        ramp_start_scale: float | None = None,
        ramp_end_scale: float | None = None,
        tf_boost: float = 1.0,
        layers: list[int] | None = None,
        branches: tuple[str, ...] = ("gdn",),
        start_step: int = 1,
    ):
        self.attenuation_scale = attenuation_scale
        self.release_step = release_step
        self.release_type = release_type
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        self.ramp_start_scale = ramp_start_scale
        self.ramp_end_scale = ramp_end_scale
        self.tf_boost = tf_boost
        self.layers = layers
        self.branches = list(branches)
        self.start_step = start_step
        if (
            attenuation_scale < 0
            or start_step < 0
            or (release_step is not None and release_step < start_step)
        ):
            raise ValueError("Invalid LR intervention range or multiplier")

    def gdn_multiplier(self, step: int) -> float:
        """Return the GDN LR multiplier at training step ``step`` (1-indexed)."""
        if step < self.start_step:
            return 1.0
        if self.release_step is None:
            return self.attenuation_scale
        if step <= self.release_step:
            if self.release_type == "ramp" and self.ramp_start and self.ramp_end:
                if step < self.ramp_start:
                    return self.attenuation_scale
                if step <= self.ramp_end:
                    frac = (step - self.ramp_start) / max(
                        1, self.ramp_end - self.ramp_start
                    )
                    s0 = (
                        self.ramp_start_scale
                        if self.ramp_start_scale is not None
                        else self.attenuation_scale
                    )
                    s1 = self.ramp_end_scale if self.ramp_end_scale is not None else 1.0
                    return s0 + frac * (s1 - s0)
                return 1.0
            return self.attenuation_scale
        return 1.0

    def apply_to_param_groups(
        self, param_groups: list[dict], base_lr: float, step: int
    ) -> None:
        """Set ``lr`` on each param group according to its ``group_name``.

        Mutates the provided ``param_groups`` in place. ``group_name`` must be
        one of ``gdn``, ``tf``, ``other`` (as produced by
        ``training.param_groups.build_param_groups``).
        """
        gdn_mult = self.gdn_multiplier(step)
        for pg in param_groups:
            gn = pg.get("group_name", "other")
            selected = (
                pg.get("layer_index") in self.layers
                if self.layers is not None
                else gn in self.branches
            )
            if selected:
                pg["lr"] = base_lr * gdn_mult
            elif gn == "tf":
                pg["lr"] = base_lr * self.tf_boost
            else:
                pg["lr"] = base_lr


class FormationGateController:
    """Per-branch forward output gain (N17 asymmetric formation gating).

    During the formation window the bottom / top GDN branch residual output is
    scaled by its configured gain (the bottom branch is soft-damped in the main
    arm). After ``release_step`` both branches return to gain=1.0. Modes:

    - "fixed": bottom/top each use their configured gain throughout the window.
    - "random": per-step choice of which single branch to damp (regularization
      control with the same expected damp amount but no asymmetric identity).
    """

    def __init__(self, config: dict, layer_indices: dict, total_steps: int):
        self.bottom_gain = config.get("bottom_gain", 1.0)
        self.top_gain = config.get("top_gain", 1.0)
        self.release_step = config.get("release_step", 3000)
        self.mode = config.get("mode", "fixed")
        self.bottom_indices = list(layer_indices.get("bottom", []))
        self.top_indices = list(layer_indices.get("top", []))
        self.total_steps = total_steps
        self._hooks = []

    def get_gains(self, step: int) -> tuple[float, float]:
        """Return ``(bottom_gain, top_gain)`` for a step (deterministic)."""
        if step > self.release_step:
            return 1.0, 1.0
        if self.mode == "random":
            import random as _random

            rng = _random.Random(step * 7919 + 42)
            if rng.random() < 0.5:
                return self.bottom_gain, 1.0
            return 1.0, self.top_gain
        return self.bottom_gain, self.top_gain

    def _make_hook(self, step_holder: dict, layer_idx: int):
        def hook(module, input, output):
            if not module.training:
                return output
            step = step_holder.get("step", 0)
            bottom_gain, top_gain = self.get_gains(step)
            if layer_idx in self.bottom_indices:
                gain = bottom_gain
            elif layer_idx in self.top_indices:
                gain = top_gain
            else:
                gain = 1.0
            if gain == 1.0:
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            # Scale the update, preserving the residual stream.
            modified = input[0] + (hidden - input[0]) * gain
            if isinstance(output, (tuple, list)):
                return (modified,) + tuple(output[1:])
            return modified

        return hook

    def patch_model(self, model, step_holder: dict):
        """Register forward hooks on the bottom + top GDN layers."""
        gdn_indices = sorted(set(self.bottom_indices) | set(self.top_indices))
        for gdn_idx in gdn_indices:
            if gdn_idx >= len(model.layers):
                continue
            layer = model.layers[gdn_idx]
            h = layer.register_forward_hook(self._make_hook(step_holder, gdn_idx))
            self._hooks.append(h)
        return model

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class Lag1AblationController:
    """Hard short-conv lag ablation (N21 lag-1/2 ablation).

    During ``[start_step, release_step]`` the GDN short-conv weight for the
    chosen lag is forced to zero (hard-disabling predecessor access). After
    ``release_step`` the zeroing stops, so the weight may re-grow from zero via
    normal gradients.

    Contrast with :class:`LRDampSchedule`: damp is *soft* (the whole GDN branch
    learns slowly, but the lag channel can still form), while this ablation is
    *hard* (the lag weight is identically zero during the window).
    """

    def __init__(self, config: dict, gdn_indices: list, total_steps: int):
        self.release_step = config.get("release_step", 3000)
        self.start_step = config.get("start_step", 0)
        self.lag = config.get("lag", 1)  # 1 = lag-1 (default), 2 = lag-2
        self.gdn_indices = list(gdn_indices)
        self.total_steps = total_steps
        self._hooks = []

    def _zero_lag(self, layer, step: int) -> None:
        if step > self.release_step or step < self.start_step:
            return
        conv_size = layer.gdn.conv_size
        # The causal short conv weight index maps to lag = conv_size - 1 - k,
        # so lag-k corresponds to index conv_size - 1 - lag.
        lag_idx = conv_size - 1 - self.lag
        if lag_idx < 0 or lag_idx >= conv_size:
            return
        with torch.no_grad():
            for m in (layer.gdn.q_conv1d, layer.gdn.k_conv1d, layer.gdn.v_conv1d):
                m.weight[..., lag_idx] = 0.0

    def _make_pre_hook(self, step_holder: dict, layer):
        def hook(module, args):
            if not module.training:
                return
            step = step_holder.get("step", 0)
            self._zero_lag(module, step)

        return hook

    def patch_model(self, model, step_holder: dict):
        """Register pre-hooks on the target GDN layers."""
        for gdn_idx in self.gdn_indices:
            if gdn_idx >= len(model.layers):
                continue
            layer = model.layers[gdn_idx]
            h = layer.register_forward_pre_hook(self._make_pre_hook(step_holder, layer))
            self._hooks.append(h)
        return model

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
