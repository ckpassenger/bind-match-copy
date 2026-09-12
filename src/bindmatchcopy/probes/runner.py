"""Paired, prompt-level measurements for controlled models.

All modes emit the actual input IDs and coordinates. These are new measurements
under the declared measurement protocol, not replacements for archived paper exports.
"""

from ..protocol import EVAL_SAMPLES


from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np
import torch

from ..interventions.attention import (
    AttentionIntervention,
    replace_source,
    suppress_source,
)
from ..interventions.update_patch import rms_match
from .base import logit_margin, record_pos_delta, margin_with_pos_patch, PosDeltaPatch


@dataclass
class PromptConfig:
    seq_len: int = 1020
    needle_min: int = 3000
    needle_max: int = 3200
    filler_min: int = 500
    filler_max: int = 3000
    replacement_min: int = 100
    replacement_max: int = 3000
    distance_min: int = 300
    distance_max: int = 700  # exclusive, matching numpy.integers in server source

    def validate(self):
        if self.seq_len < 16 or self.needle_max - self.needle_min < 4:
            raise ValueError("Prompt length/pool too small")
        if min(self.needle_min, self.filler_min, self.replacement_min) < 0:
            raise ValueError("Token IDs must be nonnegative")
        if (
            self.filler_max <= self.filler_min
            or self.replacement_max <= self.replacement_min
        ):
            raise ValueError("Empty filler or replacement pool")
        if not 3 <= self.distance_min < self.distance_max < self.seq_len - 4:
            raise ValueError("Invalid distance range")
        if not (
            self.filler_max <= self.needle_min or self.needle_max <= self.filler_min
        ):
            raise ValueError("Controlled needle and filler pools must be disjoint")
        return self


def prompt_pairs(n_samples=EVAL_SAMPLES, seed=42, config=None, lag=None):
    spec = (config or PromptConfig()).validate()
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    result = []
    for index in range(n_samples):
        a, b, c, d = map(
            int,
            rng.choice(np.arange(spec.needle_min, spec.needle_max), 4, replace=False),
        )
        distance = int(rng.integers(spec.distance_min, spec.distance_max))
        query = spec.seq_len - 1
        second = query - distance
        first = max(2, second - max(8, distance // 2))
        ids = rng.integers(spec.filler_min, spec.filler_max, spec.seq_len).tolist()
        if lag is None:
            source = first + 1
            ids[first : first + 2] = [a, b]
            ids[second : second + 2] = [c, d]
        else:
            # The lag assay keeps its distinct single-source construction.
            source = max(
                3,
                min(
                    query
                    - distance
                    - int(rng.integers(0, min(200, spec.seq_len // 5))),
                    query - 3,
                ),
            )
            first = source - lag
            if lag < 1 or first < 0:
                raise ValueError(
                    "Lag outside this prompt; increase seq_len or reduce distance"
                )
            ids[first], ids[source] = a, b
        ids[query] = a
        candidates = [
            t for t in range(spec.replacement_min, spec.replacement_max) if t != a
        ]
        if not candidates:
            raise ValueError("No replacement token distinct from predecessor")
        donor = ids.copy()
        donor[first] = int(rng.choice(candidates))
        swap = ids.copy()
        swap[first], swap[second] = c, a
        record = dict(
            index=index,
            clean_ids=ids,
            bind_donor_ids=donor,
            match_donor_ids=swap if lag is None else None,
            source_pos=source,
            predecessor_pos=first,
            query_pos=query,
            distractor_pos=second + 1 if lag is None else None,
            offsite_pos=max(0, first - 1),
            target=b,
            distractor=d,
            distance=distance,
            lag=lag,
            eval_seed=seed,
        )
        record["prompt_id"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        result.append(record)
    return result


def _validate_model(model, prompts, layers):
    if model.training:
        raise ValueError("Probes require model.eval()")
    if not layers or any(l < 0 or l >= len(model.layers) for l in layers):
        raise ValueError("Invalid or empty layer selection")
    if len(set(layers)) != len(layers):
        raise ValueError("Duplicate layers")
    config = getattr(model, "config", None)
    if config:
        for p in prompts:
            if len(p["clean_ids"]) > config.max_seq_len:
                raise ValueError("Prompt exceeds checkpoint context length")
            if (
                max(
                    p["clean_ids"]
                    + p["bind_donor_ids"]
                    + [p["target"], p["distractor"]]
                )
                >= config.vocab_size
            ):
                raise ValueError("Prompt token ID outside checkpoint vocabulary")


def _margin(model, p, device):
    return logit_margin(
        model, p["clean_ids"], p["query_pos"], p["target"], p["distractor"], device
    )


def _capture_attention(model, layer, ids, device):
    cache = {}
    with AttentionIntervention(model.layers[layer], capture=cache):
        model(torch.tensor([ids], device=device))
    return cache


@torch.no_grad()
def run_probe(
    model,
    device,
    mode,
    layers,
    *,
    n_samples=EVAL_SAMPLES,
    seed=42,
    prompt_config=None,
    lags=(1, 2, 4, 8, 16, 32, 64),
):
    if mode not in (
        "bind",
        "match",
        "lag",
        "copy",
        "selection",
        "p1",
        "head_match",
        "edge",
    ):
        raise ValueError(f"Unknown probe: {mode}")
    prompts = []
    for lag in (lags if mode == "lag" else [None]):
        prompts.extend(prompt_pairs(n_samples, seed, prompt_config, lag))
    _validate_model(model, prompts, layers)
    rows = []
    for layer in layers:
        if mode in ("copy", "selection", "p1", "head_match", "edge") and not getattr(
            model.layers[layer], "supports_attention_interventions", False
        ):
            rows.append(
                dict(
                    layer=layer,
                    status="unsupported",
                    reason="No attention P/K/V interface",
                )
            )
            continue
        for p in prompts:
            base = dict(
                prompt_id=p["prompt_id"], layer=layer, status="measured", lag=p["lag"]
            )
            clean = _margin(model, p, device)
            if mode in ("bind", "match", "lag"):
                pos = p["query_pos"] if mode == "match" else p["source_pos"]
                donor = p["match_donor_ids"] if mode == "match" else p["bind_donor_ids"]
                ref = record_pos_delta(model, layer, pos, p["clean_ids"], device)
                alt = record_pos_delta(model, layer, pos, donor, device)
                patched = margin_with_pos_patch(
                    model,
                    p["clean_ids"],
                    p["query_pos"],
                    p["target"],
                    p["distractor"],
                    device,
                    layer,
                    pos,
                    alt,
                    ref,
                )
                self_patched = margin_with_pos_patch(
                    model,
                    p["clean_ids"],
                    p["query_pos"],
                    p["target"],
                    p["distractor"],
                    device,
                    layer,
                    pos,
                    ref,
                    ref,
                )
                rows.append(
                    dict(
                        base,
                        metric="self_patch_margin",
                        m_clean=clean,
                        m_patched=self_patched,
                        value=clean - self_patched,
                        unit="logit_margin",
                    )
                )
                rows.append(
                    dict(
                        base,
                        metric=f"{mode}_update_margin",
                        m_clean=clean,
                        m_patched=patched,
                        value=clean - patched,
                        unit="logit_margin",
                    )
                )
                continue
            cc = _capture_attention(model, layer, p["clean_ids"], device)
            if mode == "edge":
                for cell, position in [
                    ("source", p["source_pos"]),
                    ("offsite", p["offsite_pos"]),
                ]:
                    with AttentionIntervention(
                        model.layers[layer],
                        pattern=suppress_source(p["query_pos"], position),
                    ):
                        patched = _margin(model, p, device)
                    rows.append(
                        dict(
                            base,
                            metric="source_edge_suppression_margin",
                            cell=cell,
                            m_clean=clean,
                            m_patched=patched,
                            value=clean - patched,
                            unit="logit_margin",
                        )
                    )
            elif mode == "p1":
                for head in range(cc["p"].shape[1]):
                    rows.append(
                        dict(
                            base,
                            metric="source_attention_mass",
                            head=head,
                            head_space="query",
                            value=float(
                                cc["p"][0, head, p["query_pos"], p["source_pos"]]
                            ),
                            unit="probability",
                        )
                    )
            elif mode == "copy":
                for head in [None] + list(range(cc["v"].shape[1])):
                    value = (
                        cc["v"][:, :, p["distractor_pos"]]
                        if head is None
                        else cc["v"][:, head, p["distractor_pos"]]
                    )
                    with AttentionIntervention(
                        model.layers[layer],
                        qkv=replace_source("v", p["source_pos"], value, head),
                        pattern=lambda _: cc["p"],
                    ):
                        patched = _margin(model, p, device)
                    rows.append(
                        dict(
                            base,
                            metric="copy_fixed_attention_margin",
                            head=head,
                            head_space="query",
                            m_clean=clean,
                            m_patched=patched,
                            value=clean - patched,
                            unit="logit_margin",
                        )
                    )
            else:
                dc = _capture_attention(model, layer, p["match_donor_ids"], device)
                conditions = (
                    [
                        ("pattern_only", dc["p"], cc["v"]),
                        ("value_only", cc["p"], dc["v"]),
                        ("joint", dc["p"], dc["v"]),
                    ]
                    if mode == "selection"
                    else [("head_match", dc["p"], dc["v"])]
                )
                for cell, pattern, values in conditions:
                    for head in (
                        [None] if mode == "selection" else range(values.shape[1])
                    ):
                        desired = (
                            pattern[:, :, p["query_pos"] : p["query_pos"] + 1].float()
                            @ values.float()
                        ).to(values.dtype)

                        def change(output):
                            output = output.clone()
                            if head is None:
                                output[:, :, p["query_pos"] : p["query_pos"] + 1] = (
                                    desired
                                )
                            else:
                                output[:, head, p["query_pos"]] = desired[:, head, 0]
                            return output

                        with AttentionIntervention(model.layers[layer], output=change):
                            patched = _margin(model, p, device)
                        rows.append(
                            dict(
                                base,
                                metric=(
                                    "selection_fixed_value_margin"
                                    if mode == "selection"
                                    else "match_head_output_margin"
                                ),
                                cell=cell,
                                head=head,
                                head_space="query",
                                m_clean=clean,
                                m_patched=patched,
                                value=clean - patched,
                                unit="logit_margin",
                            )
                        )
    return dict(
        protocol_id=f"controlled_{mode}_v31_1",
        eval_seed=seed,
        n_requested=n_samples,
        count_scope="per_layer_per_lag" if mode == "lag" else "per_layer",
        prompt_config=asdict(prompt_config or PromptConfig()),
        prompts=prompts,
        rows=rows,
    )


@torch.no_grad()
def run_route(
    model,
    device,
    sender,
    receiver,
    *,
    n_samples=EVAL_SAMPLES,
    seed=42,
    prompt_config=None,
    factorial=False,
):
    prompts = prompt_pairs(n_samples, seed, prompt_config)
    _validate_model(model, prompts, [sender, receiver])
    if not sender < receiver:
        raise ValueError("Sender must precede receiver")
    if not getattr(model.layers[receiver], "supports_attention_interventions", False):
        raise NotImplementedError("Receiver must expose attention P/K/V")
    rows = []
    for p in prompts:
        source = p["source_pos"]
        clean = _margin(model, p, device)
        ref = record_pos_delta(model, sender, source, p["clean_ids"], device)
        alt = record_pos_delta(model, sender, source, p["bind_donor_ids"], device)
        cc = _capture_attention(model, receiver, p["clean_ids"], device)
        with PosDeltaPatch(model, sender, source, alt, ref):
            perturbed = _margin(model, p, device)
            pc = _capture_attention(model, receiver, p["clean_ids"], device)
        base = dict(
            prompt_id=p["prompt_id"],
            sender_layer=sender,
            receiver_layer=receiver,
            source_pos=source,
            offsite_pos=p["offsite_pos"],
            status="measured",
            unit="logit_margin",
        )
        if factorial:
            from contextlib import nullcontext

            for bind in (0, 1):
                for edge in (0, 1):
                    for copy in (0, 1):
                        with (
                            PosDeltaPatch(model, sender, source, alt, ref)
                            if bind
                            else nullcontext()
                        ):
                            # Use the distractor V from this Bind state, not an unrelated run.
                            cache = pc if bind else cc
                            with AttentionIntervention(
                                model.layers[receiver],
                                qkv=(
                                    replace_source(
                                        "v",
                                        source,
                                        cache["v"][:, :, p["distractor_pos"]],
                                    )
                                    if copy
                                    else None
                                ),
                                pattern=(
                                    suppress_source(p["query_pos"], source)
                                    if edge
                                    else None
                                ),
                            ):
                                margin = _margin(model, p, device)
                        rows.append(
                            dict(
                                base,
                                bind=bind,
                                edge=edge,
                                copy=copy,
                                metric="factorial_cell_margin",
                                value=margin,
                            )
                        )
            continue
        rows.append(
            dict(
                base,
                cell="full_path",
                metric="path_damage_margin",
                m_clean=clean,
                m_patched=perturbed,
                value=clean - perturbed,
            )
        )
        for channel in ("k", "v", "kv"):

            def transfer(q, k, v):
                k, v = k.clone(), v.clone()
                if "k" in channel:
                    k[:, :, source] = pc["k"][:, :, source]
                if "v" in channel:
                    v[:, :, source] = pc["v"][:, :, source]
                return q, k, v

            with AttentionIntervention(model.layers[receiver], qkv=transfer):
                m = _margin(model, p, device)
            rows.append(
                dict(
                    base,
                    cell=channel,
                    metric="path_damage_margin",
                    m_clean=clean,
                    m_patched=m,
                    value=clean - m,
                )
            )
        increment = cc["k"][:, :, source] - pc["k"][:, :, source]
        for cell, pos in [
            ("source_restore", source),
            ("offsite_restore", p["offsite_pos"]),
        ]:
            transform = replace_source(
                "k",
                pos,
                cc["k"][:, :, source] if cell == "source_restore" else increment,
                additive=cell == "offsite_restore",
            )
            with PosDeltaPatch(model, sender, source, alt, ref), AttentionIntervention(
                model.layers[receiver], qkv=transform
            ):
                m = _margin(model, p, device)
            rows.append(
                dict(
                    base,
                    cell=cell,
                    metric="restoration_gain_margin",
                    m_perturbed=perturbed,
                    m_restored=m,
                    value=m - perturbed,
                    repair_l2=float(increment.float().norm()),
                )
            )
    return dict(
        protocol_id=(
            "controlled_factorial_v31_1" if factorial else "controlled_path_v31_1"
        ),
        patch_scope="receiver_source_slice_after_rope",
        n_requested=n_samples,
        eval_seed=seed,
        prompts=prompts,
        rows=rows,
    )
