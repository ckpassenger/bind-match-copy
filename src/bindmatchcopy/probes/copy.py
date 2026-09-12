"""Fixed-attention value transmission using the model's native attention path."""

from ..protocol import EVAL_SAMPLES


from .runner import run_probe


def run_copy_assay(model, device, layer_idx, n_samples=EVAL_SAMPLES, seed=31337):
    result = run_probe(
        model, device, "copy", [layer_idx], n_samples=n_samples, seed=seed
    )
    rows = result["rows"]
    if any(r["status"] != "measured" for r in rows):
        raise NotImplementedError("Copy requires attention P/V access")
    full = [r for r in rows if r["head"] is None]
    mean = lambda key: sum(r[key] for r in full) / len(full)
    heads = sorted({r["head"] for r in rows if r["head"] is not None})
    per_head = {
        h: sum(r["value"] for r in rows if r["head"] == h) / len(full) for h in heads
    }
    return (
        {
            "m_clean": mean("m_clean"),
            "m_value_to_D": mean("m_patched"),
            "U_to_D": mean("value"),
        },
        per_head,
        max(per_head, key=per_head.get),
    )
