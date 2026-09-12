import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version, PackageNotFoundError


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    """Hash executable package source, independently of user paths and git parents."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def environment():
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        source_code_sha256=code_hash(),
    )
    for name in (
        "bindmatchcopy",
        "torch",
        "numpy",
        "transformers",
        "flash-linear-attention",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_run(output, result, *, model_config, checkpoint=None, demo=False, extra=None):
    """Write samples separately from summary; prohibit silently replacing a run."""
    result = dict(result)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    from ..analysis.aggregate import summarize_rows, factorial_contrasts

    rows = result.pop("rows", [])
    prompts = result.pop("prompts", [])
    for name, records in [("rows.jsonl", rows), ("prompts.jsonl", prompts)]:
        (output / name).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
        )
    write_json(
        output / "summary.json",
        summarize_rows(rows) + summarize_rows(factorial_contrasts(rows)),
    )
    write_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            paper_version="v32",
            result_kind=(
                "demo_random_or_synthetic_model" if demo else "checkpoint_evaluation"
            ),
            checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
            model_config=model_config,
            environment=environment(),
            **result,
            **(extra or {}),
            files={
                name: sha256(output / name)
                for name in ("rows.jsonl", "prompts.jsonl", "summary.json")
            },
        ),
    )
