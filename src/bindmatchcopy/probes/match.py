"""Match probe (B circuit): does query selection follow key *content*?

The base task is ``AB ... CD ... A_q`` (target B). The key control is the
**content swap** (``CB ... AD ... A_q``): the two records' keys are swapped while
their value positions stay fixed. A content-matching layer should flip its
selection from ``B1`` to ``D1``.

The cross-architecture metric is the **query-update patch**: replace the query
position's net residual delta with the key-swapped counterpart and measure
``delta_match = m_clean - m_patch``. This is the same primitive as the Bind
probe and applies uniformly to transformer / GDN / SWA layers.
"""

from __future__ import annotations

from ..protocol import EVAL_SAMPLES

import numpy as np

from ..data.prompts import SYNTH_NEEDLE_POOL, make_pair
from .base import logit_margin, record_pos_delta, margin_with_pos_patch, query_attn_at

DIST_RANGE = (300, 700)


def run_content_swap(
    model, device, layer_idx: int, n_samples: int = EVAL_SAMPLES, seed: int = 1357
) -> dict:
    """Content-swap query-update patch (the current Match main metric)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        needle = rng.choice(SYNTH_NEEDLE_POOL, size=4, replace=False).tolist()
        clean = make_pair("clean", rng, dist, needle)
        swap = make_pair("key_swap", rng, dist, needle, base_ids=clean["ids"])
        ids, q = clean["ids"], clean["q"]
        A, B, C, D = needle

        m_clean = logit_margin(model, ids, q, B, D, device)

        clean_q_delta = record_pos_delta(model, layer_idx, q, ids, device)
        swap_q_delta = record_pos_delta(model, layer_idx, q, swap["ids"], device)
        m_patch = margin_with_pos_patch(
            model,
            ids,
            q,
            B,
            D,
            device,
            layer_idx,
            q,
            swap_q_delta,
            rms_match_to=clean_q_delta,
        )
        m_swap = logit_margin(model, swap["ids"], swap["q"], D, B, device)

        rows.append(
            {
                "m_clean": m_clean,
                "m_patch_query_swap": m_patch,
                "delta_match": m_clean - m_patch,
                "m_swap": m_swap,
            }
        )

    return {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("m_clean", "m_patch_query_swap", "delta_match", "m_swap")
    }


def run_position_swap(
    model, device, n_samples: int = EVAL_SAMPLES, seed: int = 2468
) -> float:
    """Position swap: whole records swapped, selection should still follow AB.

    Returns the mean margin ``m = z_B - z_D`` (target stays B).
    """
    rng = np.random.default_rng(seed)
    margins = []
    for _ in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        needle = rng.choice(SYNTH_NEEDLE_POOL, size=4, replace=False).tolist()
        A, B, C, D = needle
        ps = make_pair("position_swap", rng, dist, needle)
        margins.append(logit_margin(model, ps["ids"], ps["q"], B, D, device))
    return float(np.mean(margins))


def locate_match_head(
    model, device, layer_idx: int, n_samples: int = EVAL_SAMPLES, seed: int = 7777
) -> tuple[dict, int | None]:
    """Locate the match head: the head whose query attention points at ``B1``.

    Returns ``(per_head_attn_B, argmax_head)``.
    """
    rng = np.random.default_rng(seed)
    head_B: dict[int, list] = {}
    for _ in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        needle = rng.choice(SYNTH_NEEDLE_POOL, size=4, replace=False).tolist()
        clean = make_pair("clean", rng, dist, needle)
        ids, q = clean["ids"], clean["q"]
        B1 = clean["B1_pos"]
        A = query_attn_at(model, ids, q, layer_idx, device)
        if A is None:
            continue
        for h in range(A.shape[0]):
            b = A[h, q, B1].item() if B1 < A.shape[-1] else 0.0
            head_B.setdefault(h, []).append(b)

    per_head = {h: float(np.mean(head_B[h])) for h in head_B}
    if not per_head:
        return {}, None
    argmax_h = max(per_head, key=lambda k: per_head[k])
    return per_head, argmax_h
