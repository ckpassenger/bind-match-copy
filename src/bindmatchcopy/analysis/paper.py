"""Reaggregate archived figure data, without model weights or parent workspaces.

Regenerated charts are review plots, not pixel-identical paper layouts. Frozen
published assets are explicitly copied to reference/, never called new results.
"""

import csv
from importlib.resources import files
import json
from pathlib import Path
import shutil
import numpy as np

from .aggregate import mean_sd, run_mean_sd
from ..artifacts.io import sha256, write_json

NAMES = {
    "N14_baseline": "GDN / Baseline",
    "C2_prime_3K": "GDN / Early LR reduction",
    "P0_swa3_tf_base": "SWA / Baseline",
    "P0_swa3_tf_damp4K": "SWA / Early LR reduction",
    "pure_tf_8L": "Transformer",
    "N14_no_conv": "GDN / No convolution",
    "N21_gdn_skiplag1_3K": "GDN / Early lag-1 mask",
    "N21_gdn_skiplag2_3K": "GDN / Early lag-2 mask",
    "N24_hybrid_conv2": "GDN / Width-2 convolution",
    "ptf_damp_T5_gdn_matched": "Transformer / Early LR reduction",
    "350m_base": "318M / Baseline",
    "350m_botdamp_3K": "318M / G0+G1 LR reduction",
    "350m_l46damp": "318M / G1 LR reduction",
    "350m_l46l810damp": "318M / G1+G2 LR reduction",
    "350m_l14push_damp": "318M / G0+G1+G2 LR reduction",
    "early_hybrid_ih": "GDN / Enriched",
    "early_hybrid_noih": "GDN / Natural",
    "early_pure_tf_ih": "Transformer / Enriched",
    "early_pure_tf_noih": "Transformer / Natural",
    "N20_swa2": "SWA window 2",
    "N20_swa8": "SWA window 8",
    "N20_swa16": "SWA window 16",
}


def read_csv(root, name):
    with (root / "tables" / name).open() as f:
        return list(csv.DictReader(f))


def _heatmap(rows, path, title):
    import matplotlib.pyplot as plt

    labels = list(
        dict.fromkeys(
            (r["model"], r.get("seed", ""), r.get("operation", r.get("op")))
            for r in rows
        )
    )
    n = max(int(r["layer"]) for r in rows) + 1
    mat = np.full((len(labels), n), np.nan)
    lookup = {label: i for i, label in enumerate(labels)}
    for row in rows:
        if row["effect"] in ("", None):
            continue
        key = (row["model"], row.get("seed", ""), row.get("operation", row.get("op")))
        i = lookup[key]
        j = int(row["layer"])
        if np.isfinite(mat[i, j]):
            raise ValueError("Duplicate layer cell")
        mat[i, j] = float(row["effect"])
    # A shared signed scale within each figure; missing remains masked.
    scale = max(1, np.nanmax(np.abs(mat)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dedede")
    fig, ax = plt.subplots(
        figsize=(12, max(4, len(labels) * 0.24)), layout="constrained"
    )
    im = ax.imshow(
        np.ma.masked_invalid(mat), cmap=cmap, vmin=-scale, vmax=scale, aspect="auto"
    )
    ax.set_yticks(
        range(len(labels)),
        [f"{NAMES.get(m,m)} | {s} | {op}" for m, s, op in labels],
        fontsize=7,
    )
    ax.set_xticks(range(n))
    ax.set_xlabel("Layer (zero-based)")
    ax.set_title(title, loc="left", fontsize=12)
    fig.colorbar(
        im, ax=ax, label="Signed logit-margin effect; gray = unmeasured", shrink=0.65
    )
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=130)
    plt.close(fig)


def _formation(rows, path):
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["coordinate"] == "endpoint"]
    # Endpoint coordinate must remain fixed through time within each run/readout.
    keys = {}
    for r in rows:
        key = (r["model"], r["seed"], r["op"])
        keys.setdefault(key, set()).add(r["layer"])
    if any(len(v) != 1 for v in keys.values()):
        raise ValueError("Formation endpoint layer changed through time")
    summary = run_mean_sd(rows, ["model", "op", "step"])
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True)
    for col, group in enumerate(
        [
            ["N20_swa2", "N20_swa8", "N20_swa16"],
            [
                "early_hybrid_ih",
                "early_hybrid_noih",
                "early_pure_tf_ih",
                "early_pure_tf_noih",
            ],
        ]
    ):
        for row, op in enumerate(["Bind", "Match"]):
            ax = axs[row, col]
            for model in group:
                values = sorted(
                    [
                        r
                        for r in summary
                        if r["model"] == model
                        and r["op"] == op
                        and int(r["step"]) <= 8000
                    ],
                    key=lambda r: int(r["step"]),
                )
                if not values:
                    continue
                x = np.array([int(r["step"]) for r in values])
                y = np.array([r["mean"] for r in values])
                sd = np.array([r["sample_sd"] or 0 for r in values])
                (line,) = ax.plot(x, y, label=NAMES[model])
                ax.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=0.15)
            ax.set_title(f"{op}: fixed endpoint layers", loc="left")
            ax.axhline(0, color=".75", lw=0.6)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Signed margin effect")
            ax.legend(fontsize=7)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=130)
    plt.close(fig)
    return summary


def _recall(root):
    expected = read_csv(root, "recall_expected.csv")
    observations = read_csv(root, "recall_raw.csv") + read_csv(root, "recall_extra.csv")
    index = {
        (r["source"], r["model_id"], r["seed"], r["bucket"]): r for r in observations
    }
    result = []
    for row in expected:
        vals = [
            float(index[row["source"], row["model_id"], s, row["bucket"]]["ppl"])
            for s in row["training_seeds"].split("|")
        ]
        summary = mean_sd(vals)
        if not np.isclose(summary["mean"], float(row["mean"]), rtol=0, atol=1e-10):
            raise ValueError("Published recall mean mismatch")
        if not np.isclose(
            summary["sample_sd"], float(row["sample_sd"]), rtol=0, atol=1e-10
        ):
            raise ValueError("Published recall SD mismatch")
        result.append(
            dict(
                row,
                **summary,
                aggregation_unit="training_run",
                source_precision="historically_exported_per_run_ppl",
            )
        )
    return result


def reproduce_paper(output, data_root=None):
    root = (
        Path(data_root)
        if data_root
        else Path(str(files("bindmatchcopy").joinpath("paper_data/v31")))
    )
    manifest = json.loads((root / "sources.json").read_text())
    for entry in manifest["files"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Reference input hash mismatch: {entry['path']}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Reproduction output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    for source, name, title in [
        (
            "localization.csv",
            "localization_layers",
            "v31 localization: archived layer measurements",
        ),
        (
            "support.csv",
            "support_layers",
            "v31 support controls: archived layer measurements",
        ),
        ("scale318m.csv", "scale318m_layers", "v31 318M: archived layer measurements"),
    ]:
        original = read_csv(root, source)
        unique = {}
        for r in original:
            key = (r["model"], r["seed"], r["operation"], r["layer"])
            if key in unique and r["effect"] != unique[key]["effect"]:
                raise ValueError("Conflicting duplicate source cell")
            unique[key] = r
        rows = list(unique.values())
        _heatmap(rows, output / name, title)
        write_json(
            output / (name + "_aggregate.json"),
            run_mean_sd(rows, ["model", "operation", "layer"]),
        )
    qwen = read_csv(root, "qwen_fixed_token.csv")
    # Published layer scans are one archived prompt batch, not independent training runs.
    qwen = [r for r in qwen if r["op"] in ("bind", "match")]
    _heatmap(
        qwen,
        output / "qwen_fixed_token",
        "v31 fixed-token Qwen scans: archived measurements",
    )
    write_json(
        output / "formation_aggregate.json",
        _formation(read_csv(root, "formation.csv"), output / "formation"),
    )
    recall = _recall(root)
    write_json(output / "recall_table.json", recall)
    lines = [
        "# v31 recall values reconstructed from exported per-run PPL",
        "",
        "| Model ID | Bucket | Runs | PPL mean ± sample SD |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {r['model_id']} | {r['bucket']} | {r['n']} | {r['mean']:.2f} ± {r['sample_sd']:.2f} |"
        for r in recall
    )
    (output / "recall_table.md").write_text("\n".join(lines) + "\n")

    counts = sorted({r["raw_n_samples"] for r in qwen})
    report = dict(
        paper_version="v32",
        reference_files_verified=len(manifest["files"]),
        recall_cells_verified=len(recall),
        validation="means and sample SD match archived paper source to absolute tolerance 1e-10",
        scope="Review charts regenerated from archived source tables; no model inference is performed",
        paper_requested_prompts=200,
        qwen_archived_raw_n_samples=counts,
        count_status="unreconciled historical counts; no new evaluation performed",
        gaps=[
            "Fig2 lag panels are not regenerated",
            "Table1 and appendix results are not regenerated",
            "Historical per-run PPL is rounded; do not substitute exp(NLL/n) and claim unchanged paper values",
            "No trained model checkpoint or training dataset is included",
        ],
    )
    write_json(output / "reproduction_report.json", report)
    return report
