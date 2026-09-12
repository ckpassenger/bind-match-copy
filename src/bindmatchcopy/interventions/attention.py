"""Temporary interventions on native attention; no replacement block forward."""

from contextlib import AbstractContextManager


class AttentionIntervention(AbstractContextManager):
    def __init__(self, layer, *, qkv=None, pattern=None, output=None, capture=None):
        if not getattr(layer, "supports_attention_interventions", False):
            raise NotImplementedError("This layer does not expose attention P/K/V")
        self.layer = layer
        self.qkv_fn, self.pattern_fn, self.output_fn = qkv, pattern, output
        self.capture = capture

    def __enter__(self):
        if getattr(self.layer, "_bmc_attention_patch", None) is not None:
            raise RuntimeError(
                "An attention intervention is already active on this layer"
            )
        self.layer._bmc_attention_patch = self
        return self

    def __exit__(self, *exc):
        self.layer._bmc_attention_patch = None

    def qkv(self, q, k, v):
        if self.capture is not None:
            self.capture.update(
                {
                    name: t.detach().clone()
                    for name, t in zip(("q", "k", "v"), (q, k, v))
                }
            )
        return self.qkv_fn(q, k, v) if self.qkv_fn else (q, k, v)

    def pattern(self, p):
        if self.capture is not None:
            self.capture["p"] = p.detach().clone()
        return self.pattern_fn(p) if self.pattern_fn else p

    def output(self, output):
        if self.capture is not None:
            self.capture["head_output"] = output.detach().clone()
        return self.output_fn(output) if self.output_fn else output


def replace_source(channel, source, value, head=None, additive=False):
    """K/V replacement at one source, after positional transforms."""
    if channel not in ("k", "v"):
        raise ValueError("channel must be k or v")

    def transform(q, k, v):
        t = (k if channel == "k" else v).clone()
        index = (
            slice(None),
            slice(None) if head is None else head,
            source,
            slice(None),
        )
        incoming = value.to(t.device, t.dtype)
        t[index] = t[index] + incoming if additive else incoming
        return (q, t, v) if channel == "k" else (q, k, t)

    return transform


def suppress_source(query, source, dose=1.0):
    """Post-softmax source suppression and within-head renormalization (v31)."""
    if not 0 <= dose <= 1:
        raise ValueError("Edge dose must lie in [0, 1]")

    def transform(p):
        updated = p.clone()
        row = updated[:, :, query, :]
        row[:, :, source] *= 1 - dose
        denominator = row.sum(-1, keepdim=True)
        if (denominator <= 0).any():
            raise ValueError(
                "Cannot renormalize after deleting the only visible source"
            )
        updated[:, :, query, :] = row / denominator
        return updated

    return transform
