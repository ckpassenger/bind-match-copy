"""IH co-training data streams (phase-based mixture schedules).

The paper's controlled models co-train on a mixture of IH-filtered OpenWebText
and natural OpenWebText. The IH ratio is scheduled in *phases*; the canonical
schedule for the main experiments is::

    phases = [
        {"end_step": 5000,  "ih_ratio": 0.70},   # IH-heavy during formation
        {"end_step": 30000, "ih_ratio": 0.50},   # settle to 50/50 afterwards
    ]

A step belongs to the first phase whose ``end_step`` is >= step. The ratio is
the *probability* of drawing an IH paragraph for a given training step (it is
a mixture sampling ratio, **not** a paragraph-retention rate — the ``freq_rel``
filter itself keeps ~33% of paragraphs; see ``ih_filter``).

This module implements only the schedule and the mixture-sampling decision so
that the logic can be reused and unit-tested independently of any concrete
tokenizer / dataset backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class IHPhase:
    """One phase of the IH co-training schedule.

    Attributes:
        end_step: the last training step (inclusive) of this phase.
        ih_ratio: the IH sampling probability during this phase (0..1).
    """

    end_step: int
    ih_ratio: float


class IHPhaseSchedule:
    """Look up the IH mixture ratio for a given training step.

    Phases are matched in order; the first phase whose ``end_step >= step``
    wins. Steps beyond the final phase fall back to the final phase's ratio.
    """

    def __init__(self, phases: Sequence[IHPhase]):
        if not phases:
            raise ValueError("at least one phase is required")
        # Keep phases in the given (already sorted) order.
        self.phases = list(phases)
        if any(p.end_step < 1 or not 0 <= p.ih_ratio <= 1 for p in self.phases):
            raise ValueError("Invalid IH phase")
        if any(a.end_step >= b.end_step for a, b in zip(self.phases, self.phases[1:])):
            raise ValueError("IH phases must have increasing end_step values")

    @classmethod
    def natural(cls):
        return cls([IHPhase(30000, 0.0)])

    @classmethod
    def from_dicts(cls, phase_dicts: Sequence[dict]) -> "IHPhaseSchedule":
        """Build from a list of ``{"end_step": int, "ih_ratio": float}`` dicts."""
        return cls(
            [
                IHPhase(end_step=int(d["end_step"]), ih_ratio=float(d["ih_ratio"]))
                for d in phase_dicts
            ]
        )

    @classmethod
    def canonical(cls) -> "IHPhaseSchedule":
        """The canonical v31 schedule: 70% IH for steps 1..5000, then 50%."""
        return cls.from_dicts(
            [
                {"end_step": 5000, "ih_ratio": 0.70},
                {"end_step": 30000, "ih_ratio": 0.50},
            ]
        )

    def ih_ratio_at(self, step: int) -> float:
        """Return the IH mixture ratio at training step ``step`` (1-indexed)."""
        if step < 1:
            raise ValueError("Training steps are one-based")
        for phase in self.phases:
            if step <= phase.end_step:
                return phase.ih_ratio
        return self.phases[-1].ih_ratio


class MixedIHStream:
    """Sample between an IH stream and a natural (OWT) stream by phase ratio.

    The concrete IH / natural streams are injected as callables ``ih_fn()`` and
    ``natural_fn()`` returning one packed batch. ``sample_ih`` draws a Bernoulli
    with the current phase ratio; a demand-aligned training loop may inspect the
    flag to apply a per-batch damp (see ``interventions.damp``).
    """

    def __init__(self, schedule: IHPhaseSchedule, ih_fn, natural_fn, seed: int = 42):
        import numpy as np

        self.schedule = schedule
        self.ih_fn = ih_fn
        self.natural_fn = natural_fn
        self._rng = np.random.default_rng(seed)

    def sample_ih(self, step: int) -> bool:
        """Return True if this training step should draw from the IH stream."""
        return self._rng.random() < self.schedule.ih_ratio_at(step)

    def batch(self, step: int):
        """Return ``(batch, is_ih)`` for the given step."""
        is_ih = self.sample_ih(step)
        return (self.ih_fn() if is_ih else self.natural_fn()), is_ih

    def state_dict(self):
        if not all(hasattr(fn, "state_dict") for fn in (self.ih_fn, self.natural_fn)):
            raise ValueError("Exact resume requires stateful input streams")
        return {
            "rng": self._rng.bit_generator.state,
            "ih": self.ih_fn.state_dict(),
            "natural": self.natural_fn.state_dict(),
        }

    def load_state_dict(self, state):
        self._rng.bit_generator.state = state["rng"]
        self.ih_fn.load_state_dict(state["ih"])
        self.natural_fn.load_state_dict(state["natural"])
