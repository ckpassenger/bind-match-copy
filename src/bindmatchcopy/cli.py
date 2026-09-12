"""Commands separate archived-result rendering, checkpoint measurement and training."""

from .protocol import EVAL_SAMPLES, condition_counts

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def read_config(path):
    if path is None:
        return {}
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    allowed = {
        "model",
        "training",
        "lr_reduction",
        "local_head",
        "ih_phases",
        "metadata",
    }
    if set(cfg) - allowed:
        raise ValueError(f"Unknown config sections: {set(cfg)-allowed}")
    if "model" not in cfg:
        raise ValueError("Config needs a model section")
    return cfg


def _model_config(config):
    from .config import ModelConfig

    return ModelConfig.from_dict(config["model"]) if config else None


def _load_model(args, training=False):
    import torch
    from .config import build_model
    from .models.checkpoint import load_checkpoint

    conf = read_config(args.config)
    cfg = _model_config(conf)
    checkpoint = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    if checkpoint:
        model, payload = load_checkpoint(checkpoint, config=cfg, device=args.device)
        if (
            not training
            and payload.get("training_config", {}).get("demo")
            and not args.demo
        ):
            raise ValueError(
                "This is a synthetic-training checkpoint; pass --demo to label its results"
            )
        return model, payload, conf
    if not training and not args.demo:
        raise ValueError("Probe requires --checkpoint or explicit --demo")
    if cfg is None:
        raise ValueError("A fresh model requires --config")
    torch.manual_seed(args.seed)
    return build_model(cfg).to(args.device).eval(), {}, conf


def _probe(args):
    from .probes.runner import run_probe, run_route, PromptConfig
    from .artifacts.io import write_run

    model, payload, conf = _load_model(args)
    spec = (
        PromptConfig(**json.loads(Path(args.prompt_config).read_text()))
        if args.prompt_config
        else PromptConfig()
    )
    kwargs = dict(n_samples=args.n_samples, seed=args.seed, prompt_config=spec)
    if args.mode in ("path", "factorial"):
        if args.sender is None or args.receiver is None:
            raise ValueError("Path/factorial require --sender and --receiver")
        result = run_route(
            model,
            args.device,
            args.sender,
            args.receiver,
            factorial=args.mode == "factorial",
            **kwargs,
        )
    else:
        layers = (
            args.layers if args.layers is not None else list(range(len(model.layers)))
        )
        result = run_probe(
            model, args.device, args.mode, layers, lags=args.lags, **kwargs
        )
    write_run(
        args.output,
        result,
        model_config=asdict(model.config),
        checkpoint=args.checkpoint,
        demo=args.demo,
        extra=dict(
            training_step=payload.get("step"),
            training_seed=payload.get("training_config", {}).get("seed"),
            run_id=args.run_id,
            runtime_masks=payload.get("runtime_masks", {}),
            paper_requested_n=EVAL_SAMPLES,
            device=args.device,
        ),
    )
    return dict(output=args.output, mode=args.mode)


def _train(args):
    from .training.train import train_model
    from .interventions import LRDampSchedule, LagHeadScaffoldController
    from .data.streams import IHPhaseSchedule
    from .data.token_stream import PackedTokenStream

    model, payload, conf = _load_model(args, training=True)
    options = dict(conf.get("training", {}))
    options["seed"] = args.seed
    if args.steps is not None:
        options["total_steps"] = args.steps
    if args.demo and args.tokens:
        raise ValueError("--demo and --tokens are mutually exclusive")
    if not args.demo and not args.tokens:
        raise ValueError("Supply --tokens, or explicitly use --demo")
    schedule = (
        IHPhaseSchedule.from_dicts(conf["ih_phases"])
        if "ih_phases" in conf
        else IHPhaseSchedule.natural()
    )
    stream_fn = None
    if not args.demo:
        if any(p.ih_ratio > 0 for p in schedule.phases) and not args.ih_tokens:
            raise ValueError("IH mixture requires --ih-tokens")

        def stream_fn():
            opts = dict(
                batch_size=options.get("batch_size", 8),
                seq_len=options.get("seq_len", 1024),
            )
            return (
                PackedTokenStream(
                    args.ih_tokens or args.tokens, seed=args.seed + 1, **opts
                ),
                PackedTokenStream(args.tokens, seed=args.seed, **opts),
            )

    return train_model(
        model,
        model.config,
        args.device,
        args.output,
        **options,
        demo=args.demo,
        stream_fn=stream_fn,
        ih_schedule=schedule,
        resume=payload if args.resume else None,
        stop_after=args.stop_after,
        damp=(
            LRDampSchedule(**conf["lr_reduction"]) if conf.get("lr_reduction") else None
        ),
        scaffold=(
            LagHeadScaffoldController(
                conf["local_head"], options.get("total_steps", 30000)
            )
            if conf.get("local_head")
            else None
        ),
    )


def _evaluate(args):
    from .data.eval_data import load_fixed_evalset
    from .evaluation.evaluator import eval_bucket_ppl_fixed
    from .artifacts.io import write_json, sha256, environment

    model, payload, conf = _load_model(args)
    path = Path(args.evalset)
    meta = path.parent / "manifest.json"
    if not meta.is_file():
        raise ValueError(
            "Evaluation requires the adjacent manifest.json with sample counts and identity"
        )
    manifest = json.loads(meta.read_text())
    if manifest.get("evalset_sha256") != sha256(path):
        raise ValueError("Evalset hash does not match manifest")
    data = load_fixed_evalset(path)
    domains = args.domains or list(data)
    counts = condition_counts(domains, args.n_samples)
    selected = {}
    for domain, n in counts.items():
        if len(data.get(domain, [])) < n:
            raise ValueError(f"Insufficient sequences: {domain}")
        selected[domain] = data[domain][:n]
    result = eval_bucket_ppl_fixed(
        model, selected, domains, args.device, args.n_samples, model.config.max_seq_len
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new result file")
    write_json(
        output,
        dict(
            schema_version=1,
            protocol_id=manifest["protocol_id"],
            demo=args.demo,
            checkpoint_sha256=sha256(args.checkpoint),
            evalset_sha256=sha256(path),
            n_requested_per_condition=args.n_samples,
            n_actual_per_domain=counts,
            sequence_indices={d: list(range(n)) for d, n in counts.items()},
            paper_requested_sequences_per_condition=EVAL_SAMPLES,
            count_scope_note="balanced across listed domains in order; verify this allocation against original evaluation sample IDs",
            environment=environment(),
            results=result,
        ),
    )
    return dict(output=str(output))


def _qwen(args):
    from .models.adapters.qwen import load_qwen3_5_model, prepare_tokens
    from .probes.components import (
        run_component_decomposition,
        run_graded_dose,
        build_lowfreq_needle_and_filler,
        mean_agg,
        validate_receiver_parity,
        _make_pair_qwen35,
        DIST_RANGE,
    )
    from .artifacts.io import write_json, environment, sha256
    from .analysis.aggregate import paired_bootstrap

    path = Path(args.model)
    if not path.is_dir():
        raise ValueError("--model must be a pinned local Qwen3.5 checkpoint directory")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new result file")
    model, tok, cfg = load_qwen3_5_model(str(path), args.device)
    if args.tokens == "restricted":
        needle, words, filler = build_lowfreq_needle_and_filler(tok, min_id=args.min_id)
    else:
        needle, filler = prepare_tokens(tok)
    import numpy as np

    rng = np.random.default_rng(args.seed)
    first = _make_pair_qwen35(
        "clean", rng, int(rng.integers(*DIST_RANGE)), needle, filler
    )
    parity = validate_receiver_parity(model, args.receiver, first["ids"], args.device)
    fn = run_component_decomposition if args.mode == "components" else run_graded_dose
    agg, rows = fn(
        model,
        args.device,
        args.sender,
        args.receiver,
        needle,
        filler,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    positive = [r for r in rows if r["m_clean"] > 0]
    intervals = {
        k: paired_bootstrap([r[k] for r in positive])
        for k in agg
        if positive
        and ("gain" in k or "rescue_" in k)
        and k in positive[0]
        and not k.endswith("ci95")
    }
    write_json(
        output,
        dict(
            protocol_id="qwen35_components_v31_1",
            adapter_status="experimental_gpu_parity_not_verified",
            config_sha256=sha256(path / "config.json"),
            model_config=cfg.to_dict(),
            environment=environment(),
            native_receiver_parity=parity,
            patch_scope="whole_receiver_component_after_rope_for_Q_K",
            model_checkpoint_revision=args.revision,
            eval_seed=args.seed,
            n_requested=args.n_samples,
            n_actual=len(rows),
            needle_ids=needle,
            filler_ids=filler,
            mode=args.mode,
            sender=args.sender,
            receiver=args.receiver,
            all_prompts=agg,
            clean_positive=dict(
                n=len(positive),
                rule="m_clean > 0 held fixed across doses",
                means=mean_agg(positive),
                paired_ci=intervals,
            ),
            rows=rows,
        ),
    )
    return dict(
        output=str(output), adapter_status="experimental_gpu_parity_not_verified"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bmc")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "reproduce", help="Reaggregate and plot bundled v31 figure data on CPU"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--data-root")

    def reproduce(args):
        from .analysis.paper import reproduce_paper

        return reproduce_paper(args.output, args.data_root)

    p.set_defaults(func=reproduce)
    p = sub.add_parser("count-params")
    p.add_argument("--config", required=True)

    def count(args):
        from .config import build_model

        return build_model(_model_config(read_config(args.config))).count_parameters()

    p.set_defaults(func=count)
    for command, func in [
        ("probe", _probe),
        ("train", _train),
        ("evaluate", _evaluate),
    ]:
        p = sub.add_parser(command)
        p.set_defaults(func=func)
        p.add_argument("--config")
        p.add_argument("--device", default="cpu")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--demo", action="store_true")
        p.add_argument("--output", required=True)
        if command == "train":
            p.add_argument("--resume")
            p.add_argument("--tokens")
            p.add_argument("--ih-tokens")
            p.add_argument("--steps", type=int)
            p.add_argument("--stop-after", type=int)
        else:
            p.add_argument("--checkpoint", required=command == "evaluate")
            p.add_argument("--n-samples", type=int, default=EVAL_SAMPLES)
        if command == "probe":
            p.add_argument("--run-id")
            p.add_argument(
                "--mode",
                choices=[
                    "bind",
                    "match",
                    "lag",
                    "copy",
                    "selection",
                    "p1",
                    "head_match",
                    "edge",
                    "path",
                    "factorial",
                ],
                required=True,
            )
            p.add_argument("--layers", nargs="+", type=int)
            p.add_argument(
                "--lags", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64]
            )
            p.add_argument("--sender", type=int)
            p.add_argument("--receiver", type=int)
            p.add_argument("--prompt-config")
        if command == "evaluate":
            p.add_argument("--evalset", required=True)
            p.add_argument("--domains", nargs="+")
    p = sub.add_parser("prepare-eval")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-samples", type=int, default=EVAL_SAMPLES)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--demo", action="store_true")
    p.add_argument(
        "--tokenizer", help="Pinned local tokenizer, required for domain string buckets"
    )

    def prepare(args):
        from .data.eval_manifest import prepare_eval

        tok = None
        if args.tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        return prepare_eval(
            args.manifest, args.output, args.n_samples, args.seq_len, tok
        )

    p.set_defaults(func=prepare)
    p = sub.add_parser("qwen-components")
    p.set_defaults(func=_qwen)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--mode", choices=["components", "dose"], required=True)
    p.add_argument("--tokens", choices=["common", "restricted"], default="restricted")
    p.add_argument("--min-id", type=int, default=180000)
    p.add_argument("--sender", type=int, default=18)
    p.add_argument("--receiver", type=int, default=19)
    p.add_argument("--n-samples", type=int, default=EVAL_SAMPLES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (
        hasattr(args, "n_samples")
        and args.n_samples != EVAL_SAMPLES
        and not getattr(args, "demo", False)
    ):
        parser.error(
            "Paper conditions use exactly 200 samples; smaller runs require --demo"
        )
    result = args.func(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
