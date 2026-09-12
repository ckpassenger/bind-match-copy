"""Public measurement runners and explicit legacy compatibility exports."""

from .bind import run_identity_counterfactual, run_lag_profile
from .match import run_content_swap, run_position_swap, locate_match_head
from .use import run_use_assay
from .copy import run_copy_assay
from .runner import run_probe, run_route, PromptConfig, prompt_pairs
from .p1 import run_p1
from .components import (
    build_lowfreq_needle_and_filler,
    run_component_decomposition,
    run_graded_dose,
    probe_nonce_difficulty,
    mean_agg,
    bootstrap_ci,
)

__all__ = [
    "run_probe",
    "run_route",
    "PromptConfig",
    "prompt_pairs",
    "run_copy_assay",
    "run_identity_counterfactual",
    "run_lag_profile",
    "run_content_swap",
    "run_position_swap",
    "locate_match_head",
    "run_use_assay",
    "run_p1",
    "build_lowfreq_needle_and_filler",
    "run_component_decomposition",
    "run_graded_dose",
    "probe_nonce_difficulty",
    "mean_agg",
    "bootstrap_ci",
]
