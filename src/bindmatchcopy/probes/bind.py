"""Bind probe (A circuit): does position ``B1`` carry the predecessor identity?

Two complementary sub-probes:

- ``run_identity_counterfactual`` (A2): ``AB`` vs ``A'B`` with B, position,
  filler, query and distractor held fixed; patch the ``B1``-position mixer delta
  into a clean run. A strong ``delta_cf = m_clean - m_patch_cf`` plus a small
  ``delta_rescue`` proves the layer forms a Bind representation carrying A1's
  identity (position + content + causality + recoverability).
- ``run_lag_profile`` (A1): for lags d in {1,2,4,8,16,32,64}, perturb the source
  at distance ``d`` and patch the resulting ``B1`` delta. Previous-token Bind
  should be strongest at d=1 and decay with distance.

Both use the residual-delta patch primitive (see ``probes.base``), so they apply
uniformly to transformer, Gated DeltaNet, and SWA layers.
"""

from __future__ import annotations

from ..protocol import EVAL_SAMPLES

import numpy as np

from ..data.prompts import SYNTH_NEEDLE_POOL, make_pair, make_lag_sample
from .base import logit_margin, record_pos_delta, margin_with_pos_patch

LAGS = [1, 2, 4, 8, 16, 32, 64]
DIST_RANGE = (300, 700)


def run_identity_counterfactual(
    model, device, layer_idx: int, n_samples: int = EVAL_SAMPLES, seed: int = 5678
) -> tuple[dict, list]:
    """A2: AB vs A'B predecessor-identity counterfactual patch.

    Returns ``(agg, rows)`` where ``agg`` holds the mean of
    ``m_clean``, ``m_patch_cf``, ``m_rescue``, ``delta_cf``, ``delta_rescue``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        needle = rng.choice(SYNTH_NEEDLE_POOL, size=4, replace=False).tolist()
        clean = make_pair("clean", rng, dist, needle)
        cf = make_pair("counterfactual_A", rng, dist, needle, base_ids=clean["ids"])
        ids, q = clean["ids"], clean["q"]
        B1 = clean["B1_pos"]
        T, Dst = clean["T"], clean["Dst"]

        clean_delta = record_pos_delta(model, layer_idx, B1, ids, device)
        cf_delta = record_pos_delta(model, layer_idx, B1, cf["ids"], device)

        m_clean = logit_margin(model, ids, q, T, Dst, device)
        m_patch_cf = margin_with_pos_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            layer_idx,
            B1,
            cf_delta,
            rms_match_to=clean_delta,
        )
        m_rescue = margin_with_pos_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            layer_idx,
            B1,
            clean_delta,
            rms_match_to=clean_delta,
        )
        rows.append(
            {
                "m_clean": m_clean,
                "m_patch_cf": m_patch_cf,
                "m_rescue": m_rescue,
                "delta_cf": m_clean - m_patch_cf,
                "delta_rescue": m_clean - m_rescue,
            }
        )

    agg = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("m_clean", "m_patch_cf", "m_rescue", "delta_cf", "delta_rescue")
    }
    return agg, rows


def run_lag_profile(
    model, device, layer_idx: int, n_samples: int = EVAL_SAMPLES, seed: int = 1234
) -> dict:
    """A1: relative-lag profile of source influence on ``B1``.

    Returns a dict ``{str(d): mean_delta_margin}`` over d in LAGS.
    """
    rng = np.random.default_rng(seed)
    result: dict[str, float] = {}
    for d in LAGS:
        effects = []
        for _ in range(n_samples):
            dist = int(rng.integers(*DIST_RANGE))
            ids, B1, src, A, B, D, q = make_lag_sample(rng, dist, d)

            clean_delta = record_pos_delta(model, layer_idx, B1, ids, device)
            ids_cf = list(ids)
            ids_cf[src] = int(rng.integers(100, 3000))
            cf_delta = record_pos_delta(model, layer_idx, B1, ids_cf, device)

            m_clean = logit_margin(model, ids, q, B, D, device)
            m_patch = margin_with_pos_patch(
                model,
                ids,
                q,
                B,
                D,
                device,
                layer_idx,
                B1,
                cf_delta,
                rms_match_to=clean_delta,
            )
            effects.append(m_clean - m_patch)
        result[str(d)] = float(np.mean(effects))
    return result
