"""Minimal CPU example. Formal measurements are exposed through the bmc CLI."""

from .config import pure_tf_config, build_model


def run_all_smoke_tests(device="cpu"):
    import torch

    model = (
        build_model(
            pure_tf_config(
                vocab_size=128,
                d_model=16,
                n_heads=2,
                d_ff=32,
                n_layers=2,
                max_seq_len=32,
            )
        )
        .to(device)
        .eval()
    )
    logits = model(torch.arange(16, device=device)[None])["logits"]
    return bool(torch.isfinite(logits).all())
