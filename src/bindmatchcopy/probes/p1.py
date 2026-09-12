"""P1 probe: query attention mass over the historical value (induction head).

Reports ``attn_B`` (query -> B1 attention), ``attn_D`` (query -> D1), and
``attn_A`` (query -> A1). ``B_minus_D = attn_B - attn_D`` is the selectivity.
We report ``attn_B`` (not the B/A ratio) as the main score because the ratio
explodes when ``attn_A ≈ 0``.
"""

from __future__ import annotations

from ..protocol import EVAL_SAMPLES

import numpy as np

from ..data.prompts import SYNTH_NEEDLE_POOL, make_pair
from .base import query_attn_at

DIST_RANGE = (300, 700)


def run_p1(
    model, device, layer_idx: int, n_samples: int = EVAL_SAMPLES, seed: int = 42
) -> dict:
    """Measure per-head attention mass at the query of a transformer layer.

    Returns a dict mapping ``str(head_idx)`` to
    ``{"attn_B", "attn_D", "attn_A", "B_minus_D"}``.
    """
    rng = np.random.default_rng(seed)
    head_B: dict[int, list] = {}
    head_D: dict[int, list] = {}
    head_A: dict[int, list] = {}

    for _ in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        needle = rng.choice(SYNTH_NEEDLE_POOL, size=4, replace=False).tolist()
        pr = make_pair("clean", rng, dist, needle)
        ids, q = pr["ids"], pr["q"]
        B1, D1, A1 = pr["B1_pos"], pr["D1_pos"], pr["A1_pos"]
        A = query_attn_at(model, ids, q, layer_idx, device)
        if A is None:
            continue
        for h in range(A.shape[0]):
            b = A[h, q, B1].item() if B1 < A.shape[-1] else 0.0
            d = A[h, q, D1].item() if D1 < A.shape[-1] else 0.0
            a = A[h, q, A1].item() if A1 < A.shape[-1] else 0.0
            head_B.setdefault(h, []).append(b)
            head_D.setdefault(h, []).append(d)
            head_A.setdefault(h, []).append(a)

    per_head = {}
    for h in head_B:
        b = float(np.mean(head_B[h]))
        d = float(np.mean(head_D[h]))
        a = float(np.mean(head_A[h]))
        per_head[str(h)] = {
            "attn_B": b,
            "attn_D": d,
            "attn_A": a,
            "B_minus_D": b - d,
        }
    return per_head
