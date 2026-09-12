"""Block-update patching. Protocol: v31, FP32 RMS over the hidden dimension."""

from contextlib import contextmanager

import torch


def rms_match(donor, reference, eps=1e-8):
    """Match to the explicitly captured reference, including the zero case."""
    if donor.shape != reference.shape:
        raise ValueError("Donor and reference update shapes differ")
    d, r = donor.float(), reference.to(donor.device).float()
    return (d * (r.square().mean().sqrt() / (d.square().mean().sqrt() + eps))).to(
        donor.dtype
    )


@contextmanager
def capture_update(layer, pos):
    captured = {}

    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["update"] = (out - args[0])[0, pos].detach().clone()

    handle = layer.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


class UpdatePatch:
    def __init__(self, layer, pos, donor, reference=None):
        self.layer, self.pos = layer, pos
        self.donor, self.reference = donor, reference
        self.handle = None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Patch context is already active")

        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            x = args[0]
            donor = self.donor.to(x.device, x.dtype)
            if self.reference is not None:
                donor = rms_match(donor, self.reference)
            updated = out.clone()
            updated[0, self.pos] = x[0, self.pos] + donor
            return (updated,) + output[1:] if isinstance(output, tuple) else updated

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
