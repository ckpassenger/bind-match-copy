"""Needle / lag prompt construction for the lag-1 probe family.

The probe sequence follows two record templates
``... A1 B1 ... C1 D1 ... A_q ?``:

- ``A1 B1`` is the first (key, value) record; ``C1 D1`` is the distractor
  record; ``A_q`` is the repeated query key.
- The model must predict ``B1`` (the value paired with ``A1``).

Four sequence variants share the same filler background:

- ``clean``             : A B ... C D ... A_q,  target B.
- ``counterfactual_A``  : A'B ... C D ... A_q,  predecessor A changed to A'.
- ``key_swap``          : C B ... A D ... A_q,  keys swapped, values fixed.
- ``position_swap``     : C D ... A B ... A_q,  whole records swapped.

These variants are the substrate for the Bind / Match / Use / P1 probes defined
in ``bindmatchcopy.probes``.
"""

from __future__ import annotations

from typing import Any

# Synthetic token pools. The needle pool supplies the 4 distinct (A, B, C, D)
# tokens; the filler pool supplies the background. These are kept disjoint so
# the model cannot solve the task via simple co-occurrence statistics.
SYNTH_NEEDLE_POOL = list(range(3000, 3200))
SYNTH_FILLER_POOL = list(range(500, 3000))

# Fixed query position (the sequence length is 1020).
QUERY_POS = 1019


def make_pair(
    kind: str,
    rng: Any,
    dist: int,
    needle: list[int],
    base_ids: list[int] | None = None,
) -> dict:
    """Build one probe sequence (a "record pair").

    Args:
        kind: one of "clean", "counterfactual_A", "key_swap", "position_swap".
        rng: numpy random generator (must provide ``choice`` and ``integers``).
        dist: distance between the distractor key (second record) and query.
        needle: [A, B, C, D] four distinct token ids.
        base_ids: if given, apply the variant on a copy of this background so
            clean / counterfactual variants share the same filler.

    Returns:
        dict with ``ids`` and positional fields: ``A1_pos``, ``B1_pos``,
        ``C1_pos``, ``D1_pos``, ``Aq_pos``, plus ``T`` (target) and ``Dst``
        (distractor) and optionally ``A2``.
    """
    if len(needle) != 4 or len(set(needle)) != 4:
        raise ValueError("Need four distinct key/value token IDs")
    A, B, C, D = needle
    q = QUERY_POS
    r2 = q - dist
    r1 = max(2, r2 - max(8, dist // 2))
    if not 0 <= r1 < r1 + 1 < r2 < r2 + 1 < q:
        raise ValueError("Distance produces overlapping or out-of-bounds records")
    ids = (
        list(base_ids)
        if base_ids is not None
        else list(rng.choice(SYNTH_FILLER_POOL, size=1020, replace=True).tolist())
    )

    A2 = None
    if kind == "clean":
        ids[r1] = A
        ids[r1 + 1] = B
        ids[r2] = C
        ids[r2 + 1] = D
        ids[q] = A
        T, Dst = B, D
    elif kind == "counterfactual_A":
        A2 = A
        while A2 == A:
            A2 = int(rng.integers(100, 3000))
        ids[r1] = A2
        ids[r1 + 1] = B
        ids[r2] = C
        ids[r2 + 1] = D
        ids[q] = A
        T, Dst = B, D
    elif kind == "key_swap":
        ids[r1] = C
        ids[r1 + 1] = B
        ids[r2] = A
        ids[r2 + 1] = D
        ids[q] = A
        T, Dst = B, D  # keep the original target/distractor labeling
    elif kind == "position_swap":
        ids[r1] = C
        ids[r1 + 1] = D
        ids[r2] = A
        ids[r2 + 1] = B
        ids[q] = A
        T, Dst = B, D
    else:
        raise ValueError(f"Unknown pair kind: {kind}")

    out = {
        "ids": ids,
        "r1": r1,
        "r2": r2,
        "q": q,
        "A1_pos": r1,
        "B1_pos": r1 + 1,
        "C1_pos": r2,
        "D1_pos": r2 + 1,
        "Aq_pos": q,
        "T": T,
        "Dst": Dst,
    }
    if A2 is not None:
        out["A2"] = A2
    return out


def make_lag_sample(
    rng: Any, dist: int, d: int
) -> tuple[list[int], int, int, int, int, int, int]:
    """Build a source-position probe for a fixed relative lag ``d``.

    Returns ``(ids, B1_pos, src_pos, A, B, D, q)`` where ``src_pos = B1_pos - d``
    is the source whose influence on ``B1_pos`` is being measured.
    """
    needle = rng.choice(SYNTH_NEEDLE_POOL, size=4, replace=False).tolist()
    A, B, C, D = needle
    q = QUERY_POS
    B1 = q - dist - rng.integers(0, 200)
    B1 = max(3, min(B1, q - 3))
    src = B1 - d
    if d < 1 or src < 0:
        raise ValueError("Lag puts the predecessor outside the prompt")
    ids = list(rng.choice(SYNTH_FILLER_POOL, size=1020, replace=True).tolist())
    ids[src] = A
    ids[B1] = B
    ids[q] = A
    return ids, B1, src, A, B, D, q
