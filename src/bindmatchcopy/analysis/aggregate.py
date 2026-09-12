"""Keep prompt uncertainty and independent training-run variation separate."""

from collections import defaultdict
import math
import statistics
import numpy as np


def mean_sd(values):
    values = list(values)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Non-finite measurement")
    return dict(
        n=len(values),
        mean=statistics.mean(values) if values else None,
        sample_sd=statistics.stdev(values) if len(values) > 1 else None,
    )


def paired_bootstrap(differences, seed=42, n_boot=2000):
    x = np.asarray(differences, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite paired differences")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    )
    return dict(
        mean=float(x.mean()),
        ci95=np.quantile(means, [0.025, 0.975]).tolist(),
        n_pairs=len(x),
        bootstrap_seed=seed,
        n_bootstrap=n_boot,
    )


def summarize_rows(rows):
    keys = (
        "metric",
        "layer",
        "head",
        "head_space",
        "lag",
        "cell",
        "sender_layer",
        "receiver_layer",
        "bind",
        "edge",
        "copy",
    )
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "measured":
            continue
        key = tuple((k, row[k]) for k in keys if k in row)
        groups[key].append(row)
    out = []
    for key, items in groups.items():
        ids = [r.get("prompt_id") for r in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate prompt within condition {key}")
        out.append(
            dict(
                key,
                **mean_sd([r["value"] for r in items]),
                aggregation_unit="prompt",
                uncertainty="within_checkpoint_prompt_variation",
            )
        )
    return out


def run_mean_sd(rows, group_keys, value_key="effect", run_key="seed"):
    groups = defaultdict(dict)
    for row in rows:
        if row.get(value_key) in ("", None):
            continue
        key = tuple(row[k] for k in group_keys)
        run = str(row[run_key])
        if run in groups[key]:
            raise ValueError(f"Duplicate training run {run}: {key}")
        groups[key][run] = float(row[value_key])
    return [
        dict(
            zip(group_keys, key),
            **mean_sd(values.values()),
            run_ids=sorted(values),
            aggregation_unit="training_run",
        )
        for key, values in sorted(groups.items())
    ]


def factorial_contrasts(rows):
    """Paired effects at each fixed state of the other two interventions."""
    prompts = defaultdict(dict)
    for row in rows:
        if row.get("metric") != "factorial_cell_margin":
            continue
        key = (row["sender_layer"], row["receiver_layer"], row["prompt_id"])
        cell = (row["bind"], row["edge"], row["copy"])
        if cell in prompts[key]:
            raise ValueError("Duplicate factorial cell")
        prompts[key][cell] = row["value"]
    result = []
    for (sender, receiver, prompt), cells in prompts.items():
        if len(cells) != 8:
            raise ValueError("Incomplete eight-cell factorial")

        def append(cell, value):
            result.append(
                dict(
                    sender_layer=sender,
                    receiver_layer=receiver,
                    prompt_id=prompt,
                    metric="factorial_paired_contrast",
                    cell=cell,
                    value=value,
                    status="measured",
                )
            )

        for edge in (0, 1):
            for copy in (0, 1):
                append(
                    f"bind_effect_edge{edge}_copy{copy}",
                    cells[0, edge, copy] - cells[1, edge, copy],
                )
            for bind in (0, 1):
                append(
                    f"copy_effect_edge{edge}_bind{bind}",
                    cells[bind, edge, 0] - cells[bind, edge, 1],
                )
        for copy in (0, 1):
            append(
                f"bind_edge_dependence_copy{copy}",
                (cells[0, 0, copy] - cells[1, 0, copy])
                - (cells[0, 1, copy] - cells[1, 1, copy]),
            )
    return result
