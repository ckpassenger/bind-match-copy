"""Self-describing checkpoints. Legacy weights require an explicit model config."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import tempfile

import torch

from ..config import ModelConfig, build_model


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_masks(model):
    return {
        str(i): dict(layer.head_lag_limit)
        for i, layer in enumerate(model.layers)
        if getattr(layer, "head_lag_limit", None)
    }


def save_checkpoint(path, model, *, step, optimizer=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format_version=1,
        model_config=asdict(model.config),
        model_state_dict=model.state_dict(),
        step=step,
        runtime_masks=runtime_masks(model),
        **metadata
    )
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    os.close(fd)
    try:
        with open(temp, "wb") as handle:
            torch.save(payload, handle)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path, *, config=None, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    stored = payload.get("model_config")
    if stored is None and config is None:
        raise ValueError(
            "Legacy checkpoint lacks model_config; supply an explicit --config"
        )
    if stored is not None:
        if config is not None and asdict(config) != stored:
            raise ValueError("Explicit config differs from checkpoint model_config")
        config = ModelConfig.from_dict(stored)
    model = build_model(config)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    for index, mask in payload.get("runtime_masks", {}).items():
        layer = model.layers[int(index)]
        if not hasattr(layer, "head_lag_limit"):
            raise ValueError("Saved runtime mask targets an unsupported layer")
        layer.head_lag_limit = {int(h): int(lag) for h, lag in mask.items()}
    model.to(device).eval()
    return model, payload
