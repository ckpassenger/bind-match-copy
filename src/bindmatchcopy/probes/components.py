"""Experimental Qwen3.5 component restoration and perturbation-dose probes.

The sender uses a clean-RMS-matched block update. The receiver supports whole
Q/K/V components after RoPE, gate, attention output, FFN output, and full-block
controls. These are distinct from controlled-model source-slice interventions.

Restricted token IDs provide a fixed-token association condition; token rank
is not a measured corpus-frequency or novelty claim. Report clean margins and
the fixed m_clean > 0 subset, without a 50% pass threshold. Evaluate native
receiver reconstruction before using this version-specific GPU adapter.
"""

from __future__ import annotations

from ..protocol import EVAL_SAMPLES

import numpy as np
import torch

from ..models.adapters.qwen import (
    qwen35_text_layer,
    qwen35_attn_pre,
    qwen35_normal_forward,
)

DIST_RANGE = (100, 400)
QUERY_POS = 511
SEQ_LEN = 512


# ═══════════════════════════════════════════════════════════════════
# Needle / filler construction (anti-memory "nonce" pool)
# ═══════════════════════════════════════════════════════════════════


def build_lowfreq_needle_and_filler(
    tokenizer,
    needle_seed: int = 42,
    min_id: int = 150000,
    vocab_lo: int = 120000,
    vocab_hi: int = 246000,
):
    """Build a fixed 4-token needle and filler pool from restricted token IDs.

    Only ascii alphabetic single tokens of length >= 4 are kept, so the model
    cannot rely on n-gram co-occurrence priors (the anti-memory "nonce" test).

    Args:
        tokenizer: an HF tokenizer (must expose ``get_vocab`` and
            ``all_special_ids``).
        needle_seed: RNG seed selecting the 4 needle tokens.
        min_id: lower bound (inclusive) on the needle token id (difficulty knob).
        vocab_lo / vocab_hi: token-id range scanned for fixed-token candidates.

    Returns:
        ``(needle, needle_words, filler_pool)`` where ``needle`` is 4 token ids,
        ``needle_words`` the decoded strings, and ``filler_pool`` the list of
        candidate token ids (``>= min_id``).
    """
    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    inv = {v: k for k, v in vocab.items()}
    lowfreq = []
    for tid in range(vocab_lo, vocab_hi):
        if tid in special_ids:
            continue
        s = inv.get(tid, "")
        clean = s.replace("\u0120", "").replace("\u2581", "")
        if clean.isascii() and clean.isalpha() and len(clean) >= 4:
            lowfreq.append(tid)
    pool = [tid for tid in lowfreq if tid >= min_id]
    rng = np.random.default_rng(needle_seed)
    needle = rng.choice(pool, size=4, replace=False).tolist()
    needle_words = [inv.get(t, "").replace("\u0120", "") for t in needle]
    return needle, needle_words, pool


def _make_pair_qwen35(
    kind: str, rng, dist: int, needle: list[int], filler_pool: list[int], base_ids=None
) -> dict:
    """Build a Qwen3.5 probe sequence (512 tokens, query at position 511)."""
    if len(needle) != 4 or len(set(needle)) != 4 or not filler_pool:
        raise ValueError(
            "Four distinct needles and a nonempty filler pool are required"
        )
    if not any(t != needle[0] for t in filler_pool):
        raise ValueError("Filler pool has no predecessor replacement")
    if base_ids is not None and len(base_ids) != SEQ_LEN:
        raise ValueError("Paired donor length mismatch")
    A, B, C, D = needle
    q = QUERY_POS
    r2 = q - dist
    r1 = max(2, r2 - max(8, dist // 2))
    ids = (
        list(base_ids)
        if base_ids is not None
        else rng.choice(filler_pool, size=SEQ_LEN, replace=True).tolist()
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
            A2 = int(rng.choice(filler_pool))
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
        T, Dst = B, D
    elif kind == "position_swap":
        ids[r1] = C
        ids[r1 + 1] = D
        ids[r2] = A
        ids[r2 + 1] = B
        ids[q] = A
        T, Dst = B, D
    else:
        raise ValueError(kind)
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


# ═══════════════════════════════════════════════════════════════════
# Qwen3.5 forward primitives (multimodal forward needs pixel_values=None, use_cache=False)
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def _logit_margin_qwen35(
    model, ids: list[int], qpos: int, T: int, D: int, device
) -> float:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=x, pixel_values=None, use_cache=False)
    logits = out.logits[0, qpos, :].float()
    return (logits[T] - logits[D]).item()


@torch.no_grad()
def _record_pos_delta_qwen35(
    model, layer_idx: int, pos: int, ids: list[int], device
) -> torch.Tensor:
    from ..interventions.update_patch import capture_update

    with capture_update(qwen35_text_layer(model, layer_idx), pos) as captured:
        model(
            input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            pixel_values=None,
            use_cache=False,
        )
    return captured["update"]


from ..interventions.update_patch import UpdatePatch, rms_match


class _PosDeltaPatchQwen35(UpdatePatch):
    def __init__(self, model, layer_idx, pos, cf_delta, rms_match_to=None):
        super().__init__(
            qwen35_text_layer(model, layer_idx), pos, cf_delta, rms_match_to
        )


@torch.no_grad()
def _margin_with_pos_patch_qwen35(
    model,
    ids,
    qpos,
    T,
    D,
    device,
    layer_idx,
    pos,
    cf_delta,
    rms_match_to=None,
) -> float:
    with _PosDeltaPatchQwen35(
        model, layer_idx, pos, cf_delta, rms_match_to=rms_match_to
    ):
        return _logit_margin_qwen35(model, ids, qpos, T, D, device)


# ═══════════════════════════════════════════════════════════════════
# Component capture / patch (Q/K/V/gate, attn_out, ffn_out, short)
# ═══════════════════════════════════════════════════════════════════


def _make_attn_forward(attn, mode=None, targets=None):
    """Rewrite ``self_attn.forward`` for Q/K/V/gate injection or capture."""
    _a = attn

    def fn(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        q, k, v, input_shape, gate = qwen35_attn_pre(
            _a, hidden_states, position_embeddings
        )
        if mode == "q":
            q = targets["Q"].to(q.device, q.dtype)
        elif mode == "k":
            k = targets["K"].to(k.device, k.dtype)
        elif mode == "v":
            v = targets["V"].to(v.device, v.dtype)
        elif mode == "gate":
            gate = targets["gate"].to(gate.device, gate.dtype)
        out, _ = qwen35_normal_forward(_a, q, k, v, input_shape, gate)
        if mode == "capture":
            targets["Q"] = q.detach().clone()
            targets["K"] = k.detach().clone()
            targets["V"] = v.detach().clone()
            targets["gate"] = gate.detach().clone()
        return out, None

    return fn


def _make_layer_forward(layer, mode=None, targets=None):
    """Rewrite ``DecoderLayer.forward`` for attn/ffn injection, short, capture."""
    _l = layer

    def fn(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        if mode == "short":
            return hidden_states
        residual = hidden_states
        hidden_states = _l.input_layernorm(hidden_states)
        if _l.layer_type == "linear_attention":
            attn_out = _l.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            attn_out, _ = _l.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if mode == "attn":
            attn_out = targets["attn_out"].to(attn_out.device, attn_out.dtype)
        elif mode == "attn_zero":
            attn_out = torch.zeros_like(attn_out)
        hidden_states = residual + attn_out

        residual2 = hidden_states
        hidden_states = _l.post_attention_layernorm(hidden_states)
        ffn_out = _l.mlp(hidden_states)
        if mode == "ffn":
            ffn_out = targets["ffn_out"].to(ffn_out.device, ffn_out.dtype)
        elif mode == "ffn_zero":
            ffn_out = torch.zeros_like(ffn_out)
        if mode == "capture":
            targets["attn_out"] = attn_out.detach().clone()
            targets["ffn_out"] = ffn_out.detach().clone()
        hidden_states = residual2 + ffn_out
        return hidden_states

    return fn


@torch.no_grad()
def _capture_components(model, receiver: int, ids: list[int], device) -> dict:
    """One forward pass captures Q/K/V/gate/attn_out/ffn_out at the receiver."""
    layer = qwen35_text_layer(model, receiver)
    attn = layer.self_attn
    targets: dict = {}
    orig_attn = attn.forward
    orig_layer = layer.forward
    attn.forward = _make_attn_forward(attn, mode="capture", targets=targets)
    layer.forward = _make_layer_forward(layer, mode="capture", targets=targets)
    try:
        model(
            input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            pixel_values=None,
            use_cache=False,
        )
    finally:
        attn.forward = orig_attn
        layer.forward = orig_layer
    return targets


@torch.no_grad()
def _margin_with_component_patch(
    model,
    ids,
    qpos,
    T,
    D,
    device,
    sender,
    B1,
    cf_delta,
    clean_delta,
    receiver,
    mode,
    targets,
) -> float:
    """Patch sender + inject a receiver component, return the logit margin."""
    layer = qwen35_text_layer(model, receiver)
    attn = layer.self_attn
    orig_attn = attn.forward
    orig_layer = layer.forward
    if mode in ("q", "k", "v", "gate"):
        attn.forward = _make_attn_forward(attn, mode=mode, targets=targets)
    elif mode in ("attn", "ffn", "short", "attn_zero", "ffn_zero"):
        layer.forward = _make_layer_forward(layer, mode=mode, targets=targets)
    try:
        with _PosDeltaPatchQwen35(
            model, sender, B1, cf_delta, rms_match_to=clean_delta
        ):
            return _logit_margin_qwen35(model, ids, qpos, T, D, device)
    finally:
        attn.forward = orig_attn
        layer.forward = orig_layer


# ═══════════════════════════════════════════════════════════════════
# Row aggregation (pure function, unit-testable without a model)
# ═══════════════════════════════════════════════════════════════════


def mean_agg(rows: list[dict]) -> dict:
    """Mean-aggregate a list of per-sample dicts into a single dict."""
    if not rows:
        return {}
    return {
        k: float(np.mean([r[k] for r in rows]))
        for k, v in rows[0].items()
        if isinstance(v, (int, float, np.number)) and k != "prompt_index"
    }


def bootstrap_ci(vals, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Return the 95% bootstrap CI of the mean of ``vals``."""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        means.append(vals[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ═══════════════════════════════════════════════════════════════════
# Component decomposition
# ═══════════════════════════════════════════════════════════════════


def run_component_decomposition(
    model,
    device,
    sender: int,
    receiver: int,
    needle: list[int],
    filler_pool: list[int],
    n_samples: int = EVAL_SAMPLES,
    seed: int = 42,
) -> tuple[dict, list]:
    """Decompose the sender->receiver margin drop into receiver components.

    Args:
        model: a loaded Qwen3.5 model (from
            ``bindmatchcopy.models.adapters.qwen.load_qwen3_5_model``).
        device: torch device.
        sender: sender layer index (linear Bind layer, e.g. 18).
        receiver: receiver layer index (full Match/Use layer, e.g. 19).
        needle: 4 token ids [A, B, C, D].
        filler_pool: filler token ids.
        n_samples / seed: number of pairs and RNG seed.

    Returns:
        ``(agg, rows)``. ``agg`` holds means of ``m_clean``, ``m_full_path``,
        ``rescue_gain_{Q,K,V,gate,attn,ffn}``, ``short_gain``,
        ``attn_zero_gain``, ``ffn_zero_gain``, and ``delta_full``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for prompt_index in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        clean = _make_pair_qwen35("clean", rng, dist, needle, filler_pool)
        cf = _make_pair_qwen35(
            "counterfactual_A", rng, dist, needle, filler_pool, base_ids=clean["ids"]
        )
        ids, q = clean["ids"], clean["q"]
        B1 = clean["B1_pos"]
        T, Dst = clean["T"], clean["Dst"]

        clean_delta = _record_pos_delta_qwen35(model, sender, B1, ids, device)
        cf_delta = _record_pos_delta_qwen35(model, sender, B1, cf["ids"], device)

        m_clean = _logit_margin_qwen35(model, ids, q, T, Dst, device)
        m_full_path = _margin_with_pos_patch_qwen35(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            rms_match_to=clean_delta,
        )

        comps = _capture_components(model, receiver, ids, device)

        m_rescue_Q = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "q",
            comps,
        )
        m_rescue_K = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "k",
            comps,
        )
        m_rescue_V = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "v",
            comps,
        )
        m_rescue_gate = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "gate",
            comps,
        )
        m_rescue_attn = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "attn",
            comps,
        )
        m_rescue_ffn = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "ffn",
            comps,
        )
        m_short = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "short",
            {},
        )
        m_attn_zero = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "attn_zero",
            {},
        )
        m_ffn_zero = _margin_with_component_patch(
            model,
            ids,
            q,
            T,
            Dst,
            device,
            sender,
            B1,
            cf_delta,
            clean_delta,
            receiver,
            "ffn_zero",
            {},
        )

        rows.append(
            {
                "prompt_index": prompt_index,
                "clean_ids": ids,
                "donor_ids": cf["ids"],
                "source_pos": B1,
                "query_pos": q,
                "target": T,
                "distractor": Dst,
                "m_clean": m_clean,
                "m_full_path": m_full_path,
                "m_rescue_Q": m_rescue_Q,
                "m_rescue_K": m_rescue_K,
                "m_rescue_V": m_rescue_V,
                "m_rescue_gate": m_rescue_gate,
                "m_rescue_attn": m_rescue_attn,
                "m_rescue_ffn": m_rescue_ffn,
                "m_short": m_short,
                "m_attn_zero": m_attn_zero,
                "m_ffn_zero": m_ffn_zero,
                "delta_full": m_clean - m_full_path,
                "rescue_gain_Q": m_rescue_Q - m_full_path,
                "rescue_gain_K": m_rescue_K - m_full_path,
                "rescue_gain_V": m_rescue_V - m_full_path,
                "rescue_gain_gate": m_rescue_gate - m_full_path,
                "rescue_gain_attn": m_rescue_attn - m_full_path,
                "rescue_gain_ffn": m_rescue_ffn - m_full_path,
                "short_gain": m_short - m_full_path,
                "attn_zero_gain": m_attn_zero - m_full_path,
                "ffn_zero_gain": m_ffn_zero - m_full_path,
            }
        )

    return mean_agg(rows), rows


# ═══════════════════════════════════════════════════════════════════
# Graded dose (alpha-mixed sender delta) — floor-confound control
# ═══════════════════════════════════════════════════════════════════


class _MixedDeltaPatchQwen35(UpdatePatch):
    """Interpolate first, then RMS match to the recorded clean reference."""

    def __init__(
        self, model, layer_idx, pos, clean_delta, cf_delta, alpha, rms_match_to=None
    ):
        if not 0 <= alpha <= 1:
            raise ValueError("Dose must lie in [0, 1]")
        mixed = (1 - alpha) * clean_delta + alpha * cf_delta
        super().__init__(qwen35_text_layer(model, layer_idx), pos, mixed, rms_match_to)


@torch.no_grad()
def _margin_with_mixed_patch(
    model,
    ids,
    qpos,
    T,
    D,
    device,
    sender,
    B1,
    clean_delta,
    cf_delta,
    alpha,
    rms_match_to=None,
) -> float:
    with _MixedDeltaPatchQwen35(
        model,
        sender,
        B1,
        clean_delta,
        cf_delta,
        alpha,
        rms_match_to=rms_match_to,
    ):
        return _logit_margin_qwen35(model, ids, qpos, T, D, device)


@torch.no_grad()
def _margin_with_component_patch_dose(
    model,
    ids,
    qpos,
    T,
    D,
    device,
    sender,
    B1,
    clean_delta,
    cf_delta,
    alpha,
    receiver,
    mode,
    targets,
) -> float:
    layer = qwen35_text_layer(model, receiver)
    attn = layer.self_attn
    orig_attn = attn.forward
    orig_layer = layer.forward
    if mode in ("q", "k", "v", "gate"):
        attn.forward = _make_attn_forward(attn, mode=mode, targets=targets)
    elif mode in ("attn", "ffn", "short", "attn_zero", "ffn_zero"):
        layer.forward = _make_layer_forward(layer, mode=mode, targets=targets)
    try:
        with _MixedDeltaPatchQwen35(
            model,
            sender,
            B1,
            clean_delta,
            cf_delta,
            alpha,
            rms_match_to=clean_delta,
        ):
            return _logit_margin_qwen35(model, ids, qpos, T, D, device)
    finally:
        attn.forward = orig_attn
        layer.forward = orig_layer


def run_graded_dose(
    model,
    device,
    sender: int,
    receiver: int,
    needle: list[int],
    filler_pool: list[int],
    doses: list[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_samples: int = EVAL_SAMPLES,
    seed: int = 42,
) -> tuple[dict, list]:
    """Dose-response curve of the sender patch (clean -> counterfactual).

    For each alpha, records ``m_patch_a{alpha}``, ``rescue_K_a{alpha}``,
    ``rescue_attn_a{alpha}`` and ``delta_full_a{alpha}``. A non-zero rescue gain
    at low alpha (task still solvable) rules out the floor confound.

    Returns ``(agg, rows)`` where ``agg`` includes per-alpha means and 95% CI
    for the rescue gains and delta_full.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for prompt_index in range(n_samples):
        dist = int(rng.integers(*DIST_RANGE))
        clean = _make_pair_qwen35("clean", rng, dist, needle, filler_pool)
        cf = _make_pair_qwen35(
            "counterfactual_A", rng, dist, needle, filler_pool, base_ids=clean["ids"]
        )
        ids, q = clean["ids"], clean["q"]
        B1 = clean["B1_pos"]
        T, Dst = clean["T"], clean["Dst"]

        clean_delta = _record_pos_delta_qwen35(model, sender, B1, ids, device)
        cf_delta = _record_pos_delta_qwen35(model, sender, B1, cf["ids"], device)
        m_clean = _logit_margin_qwen35(model, ids, q, T, Dst, device)
        comps = _capture_components(model, receiver, ids, device)

        row = {
            "prompt_index": prompt_index,
            "clean_ids": ids,
            "donor_ids": cf["ids"],
            "source_pos": B1,
            "query_pos": q,
            "target": T,
            "distractor": Dst,
            "m_clean": m_clean,
        }
        for a in doses:
            mixed = (1 - a) * clean_delta + a * cf_delta
            effective = rms_match(mixed, clean_delta)
            row[f"patch_l2_a{a}"] = float(
                (effective.float() - clean_delta.float()).norm()
            )
            m_patch = _margin_with_mixed_patch(
                model,
                ids,
                q,
                T,
                Dst,
                device,
                sender,
                B1,
                clean_delta,
                cf_delta,
                a,
                rms_match_to=clean_delta,
            )
            m_rescue_K = _margin_with_component_patch_dose(
                model,
                ids,
                q,
                T,
                Dst,
                device,
                sender,
                B1,
                clean_delta,
                cf_delta,
                a,
                receiver,
                "k",
                comps,
            )
            m_rescue_attn = _margin_with_component_patch_dose(
                model,
                ids,
                q,
                T,
                Dst,
                device,
                sender,
                B1,
                clean_delta,
                cf_delta,
                a,
                receiver,
                "attn",
                comps,
            )
            row[f"m_patch_a{a}"] = m_patch
            row[f"rescue_K_a{a}"] = m_rescue_K - m_patch
            row[f"rescue_attn_a{a}"] = m_rescue_attn - m_patch
            row[f"delta_full_a{a}"] = m_clean - m_patch
        rows.append(row)

    agg = mean_agg(rows)
    for a in doses:
        for key in (f"rescue_K_a{a}", f"rescue_attn_a{a}", f"delta_full_a{a}"):
            vals = np.array([r[key] for r in rows], dtype=float)
            agg[f"{key}_ci95"] = bootstrap_ci(vals)
    return agg, rows


# ═══════════════════════════════════════════════════════════════════
# Nonce needle difficulty sweep (clean-solvability gate calibration)
# ═══════════════════════════════════════════════════════════════════


def probe_nonce_difficulty(
    model,
    tokenizer,
    device,
    min_ids: list[int] = (120000, 150000, 180000, 200000, 220000, 240000),
    n_samples: int = EVAL_SAMPLES,
    seed: int = 42,
) -> list[dict]:
    """Sweep needle difficulty and report the clean-margin distribution.

    For each ``min_id``, builds a restricted-ID needle, then measures the clean
    logit margin on ``n_samples`` pairs. All pools are reported; no 50% acceptance threshold is imposed.
    Token ID is an input-pool definition, not measured corpus frequency.

    Returns a list of per-``min_id`` records with ``mean``, ``median``, ``std``,
    ``min``, ``max``, ``pos_ratio``, ``n_pos``, ``needle`` and ``pool_size``.
    """
    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    inv = {v: k for k, v in vocab.items()}
    lowfreq = []
    for tid in range(120000, 246000):
        if tid in special_ids:
            continue
        s = inv.get(tid, "")
        clean = s.replace("\u0120", "").replace("\u2581", "")
        if clean.isascii() and clean.isalpha() and len(clean) >= 4:
            lowfreq.append(tid)

    results = []
    for min_id in min_ids:
        pool = [t for t in lowfreq if t >= min_id]
        needle = np.random.default_rng(42).choice(pool, size=4, replace=False).tolist()
        words = [inv.get(t, "").replace("\u0120", "") for t in needle]

        rng = np.random.default_rng(seed)
        margins = []
        for prompt_index in range(n_samples):
            dist = int(rng.integers(*DIST_RANGE))
            pair = _make_pair_qwen35("clean", rng, dist, needle, pool)
            margins.append(
                _logit_margin_qwen35(
                    model, pair["ids"], pair["q"], pair["T"], pair["Dst"], device
                )
            )
        margins = np.array(margins, dtype=float)
        results.append(
            {
                "min_id": min_id,
                "pool_size": len(pool),
                "needle": [{"word": w, "id": t} for w, t in zip(words, needle)],
                "n": n_samples,
                "m_clean_mean": round(float(margins.mean()), 5),
                "m_clean_median": round(float(np.median(margins)), 5),
                "m_clean_std": round(float(margins.std()), 5),
                "m_clean_min": round(float(margins.min()), 5),
                "m_clean_max": round(float(margins.max()), 5),
                "pos_ratio": round(float((margins > 0).mean()), 4),
                "n_pos": int((margins > 0).sum()),
            }
        )
    return results


@torch.no_grad()
def validate_receiver_parity(model, receiver, ids, device, atol=0.05, rtol=0.01):
    """Check identity reconstruction against the loaded native decoder before probes."""
    x = torch.tensor([ids], device=device)
    layer = qwen35_text_layer(model, receiver)
    if layer.layer_type != "full_attention":
        raise ValueError("Receiver must be a full-attention layer")
    native = (
        model(input_ids=x, pixel_values=None, use_cache=False).logits[:, -1].float()
    )
    old_layer, old_attn = layer.forward, layer.self_attn.forward
    try:
        layer.forward = _make_layer_forward(layer)
        layer.self_attn.forward = _make_attn_forward(layer.self_attn)
        reconstructed = (
            model(input_ids=x, pixel_values=None, use_cache=False).logits[:, -1].float()
        )
    finally:
        layer.forward = old_layer
        layer.self_attn.forward = old_attn
    torch.testing.assert_close(reconstructed, native, atol=atol, rtol=rtol)
    return dict(
        max_abs_error=float((reconstructed - native).abs().max()),
        atol=atol,
        rtol=rtol,
        scope="receiver identity reconstruction at first controlled prompt; not full server parity",
    )
