"""Formation-time interventions (lag mask, damp, gating, ablation)."""

from .mask import apply_lag_mask, LagHeadScaffoldController
from .damp import (
    LRDampSchedule,
    FormationGateController,
    Lag1AblationController,
)

__all__ = [
    "apply_lag_mask",
    "LagHeadScaffoldController",
    "LRDampSchedule",
    "FormationGateController",
    "Lag1AblationController",
]
