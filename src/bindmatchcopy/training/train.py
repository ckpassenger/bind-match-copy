"""Resumable single-device trainer for local packed tokens or explicit demos.

This is not the original multi-GPU training harness. See docs/VALIDATION.md.
"""

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch

from .param_groups import build_param_groups
from ..data.streams import IHPhaseSchedule, MixedIHStream
from ..data.token_stream import SyntheticTokenStream
from ..models.checkpoint import save_checkpoint, runtime_masks


def make_synthetic_streams(vocab_size, seq_len, batch_size, seed=0):
    return (
        SyntheticTokenStream(vocab_size, seq_len, batch_size, seed + 1, True),
        SyntheticTokenStream(vocab_size, seq_len, batch_size, seed, False),
    )


def _cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=3e-5):
    """One-based updates: warmup ends at base_lr, final update reaches min_lr."""
    if not 1 <= step <= total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError("Invalid step or warmup schedule")
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train_model(
    model,
    config,
    device,
    output_dir,
    *,
    total_steps=30000,
    batch_size=8,
    seq_len=1024,
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    seed=42,
    warmup_steps=1000,
    grad_accum=8,
    grad_clip=1.0,
    log_every=50,
    ckpt_every=1000,
    ih_schedule=None,
    damp=None,
    gate=None,
    scaffold=None,
    lag_ablation=None,
    stream_fn=None,
    demo=False,
    precision="fp32",
    resume=None,
    stop_after=None,
):
    """Train with stateful ``stream_fn() -> (IH_callable, natural_callable)``.

    Seed before constructing a fresh model (the CLI does this). Resume accepts
    a payload from load_checkpoint, with model weights already loaded. Changing
    training parameters, streams, schedule or intervention on resume is rejected.
    ``stop_after`` permits interruption without changing the planned LR curve.
    """
    device = torch.device(device)
    if (
        min(total_steps, batch_size, grad_accum, log_every, ckpt_every) < 1
        or seq_len < 2
    ):
        raise ValueError("Positive training counts and seq_len >= 2 required")
    if seq_len > config.max_seq_len or not 0 <= min_lr <= lr or grad_clip <= 0:
        raise ValueError("Invalid context length, LR floor or clipping norm")
    if precision not in ("fp32", "bf16") or (
        precision == "bf16" and device.type != "cuda"
    ):
        raise ValueError("Supported precision: fp32, or bf16 on CUDA")
    _cosine_lr(1, lr, warmup_steps, total_steps, min_lr)
    if stream_fn is None and not demo:
        raise ValueError("Supply local token streams, or explicitly set demo=True")
    if stream_fn is not None and demo:
        raise ValueError("A demo cannot also use real token streams")
    if gate or lag_ablation:
        raise NotImplementedError(
            "Legacy gate/conv-ablation controllers require server validation; not enabled in the reproducible trainer"
        )
    streams = (
        make_synthetic_streams(config.vocab_size, seq_len, batch_size, seed)
        if demo
        else stream_fn()
    )
    schedule = ih_schedule or IHPhaseSchedule.natural()
    stream = MixedIHStream(schedule, *streams, seed=seed)
    stream.state_dict()  # Fail before training if exact resume is unavailable.
    intervention = dict(
        lr_reduction=dict(vars(damp)) if damp else None,
        local_head=(
            {k: v for k, v in vars(scaffold).items() if not k.startswith("_")}
            if scaffold
            else None
        ),
    )
    spec = dict(
        total_steps=total_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        seed=seed,
        warmup_steps=warmup_steps,
        grad_accum=grad_accum,
        grad_clip=grad_clip,
        precision=precision,
        demo=demo,
        ih_phases=[asdict(p) for p in schedule.phases],
        intervention=intervention,
    )
    param_groups = build_param_groups(model, config.layer_types, weight_decay)
    optim = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    start = 0
    if resume is not None:
        if resume.get("training_config") != spec:
            raise ValueError("Resume training_config mismatch")
        for key in ("optimizer_state_dict", "data_state", "rng_state", "step"):
            if key not in resume:
                raise ValueError(f"Cannot resume: missing {key}")
        optim.load_state_dict(resume["optimizer_state_dict"])
        stream.load_state_dict(resume["data_state"])
        torch.set_rng_state(resume["rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        start = resume["step"]
    else:
        torch.manual_seed(seed)
    end = total_steps if stop_after is None else min(stop_after, total_steps)
    if not start < end:
        raise ValueError("No remaining updates requested")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "train_log.jsonl").exists():
        raise FileExistsError(
            "Output already contains a training log; choose a new directory or resume"
        )
    (output / "training_config.json").write_text(json.dumps(spec, indent=2) + "\n")
    baseline_masks = (
        resume.get("baseline_runtime_masks", {}) if resume else runtime_masks(model)
    )
    if resume:
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "head_lag_limit"):
                layer.head_lag_limit = baseline_masks.get(str(i), None)
    step_holder = {"step": start}
    final_loss = None
    try:
        if scaffold:
            scaffold.patch_model(model, step_holder)
        with (output / "train_log.jsonl").open("a" if resume else "w") as log:
            for step in range(start + 1, end + 1):
                model.train()
                step_holder["step"] = step
                base_lr = _cosine_lr(step, lr, warmup_steps, total_steps, min_lr)
                # Use optimizer-owned groups: load_state_dict replaces its dictionaries.
                if damp:
                    damp.apply_to_param_groups(optim.param_groups, base_lr, step)
                else:
                    for pg in optim.param_groups:
                        pg["lr"] = base_lr
                optim.zero_grad(set_to_none=True)
                final_loss = 0.0
                ih_batches = 0
                for _ in range(grad_accum):
                    tokens, is_ih = stream.batch(step)
                    tokens = tokens.to(device)
                    if (
                        tokens.shape != (batch_size, seq_len)
                        or tokens.min() < 0
                        or tokens.max() >= config.vocab_size
                    ):
                        raise ValueError(
                            "Input batch shape or vocabulary does not match training config"
                        )
                    ctx = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if precision == "bf16"
                        else nullcontext()
                    )
                    with ctx:
                        loss = model(tokens, labels=tokens)["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    (loss / grad_accum).backward()
                    final_loss += float(loss.detach()) / grad_accum
                    ih_batches += int(is_ih)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True
                )
                optim.step()
                if step % log_every == 0 or step == end:
                    log.write(
                        json.dumps(
                            dict(
                                step=step,
                                loss=final_loss,
                                lr=base_lr,
                                ih_batches=ih_batches,
                                effective_batch_size=batch_size * grad_accum,
                            )
                        )
                        + "\n"
                    )
                    log.flush()
                if step % ckpt_every == 0 or step == end:
                    metadata = dict(
                        training_config=spec,
                        data_state=stream.state_dict(),
                        baseline_runtime_masks=baseline_masks,
                        rng_state=torch.get_rng_state(),
                        loss=final_loss,
                        cuda_rng_state=(
                            torch.cuda.get_rng_state_all()
                            if device.type == "cuda"
                            else []
                        ),
                    )
                    path = output / (
                        "checkpoint_final.pt"
                        if step == end
                        else f"checkpoint_step{step}.pt"
                    )
                    save_checkpoint(path, model, step=step, optimizer=optim, **metadata)
    finally:
        if scaffold:
            scaffold.remove_hooks()
    return dict(
        step=end,
        loss=final_loss,
        checkpoint=str(output / "checkpoint_final.pt"),
        demo=demo,
    )
