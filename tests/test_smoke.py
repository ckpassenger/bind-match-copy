"""Small public smoke test. Research conditions use 200; this checks mechanics."""

import torch
from bindmatchcopy.config import pure_tf_config, build_model
from bindmatchcopy.protocol import EVAL_SAMPLES, condition_counts
from bindmatchcopy.probes.runner import PromptConfig, run_probe


def test_smoke():
    assert EVAL_SAMPLES == 200
    assert sum(condition_counts(["wiki", "python", "math"]).values()) == 200
    cfg = pure_tf_config(
        vocab_size=128, d_model=16, n_heads=2, d_ff=32, n_layers=2, max_seq_len=64
    )
    torch.manual_seed(42)
    model = build_model(cfg).eval()
    prompts = PromptConfig(
        seq_len=32,
        needle_min=64,
        needle_max=80,
        filler_min=0,
        filler_max=60,
        replacement_min=0,
        replacement_max=60,
        distance_min=8,
        distance_max=16,
    )
    result = run_probe(model, "cpu", "bind", [0], n_samples=2, prompt_config=prompts)
    assert len(result["prompts"]) == 2
    assert all(torch.isfinite(torch.tensor(row["value"])) for row in result["rows"])
