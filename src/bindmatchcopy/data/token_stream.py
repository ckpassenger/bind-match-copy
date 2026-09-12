"""Stateful local token streams; no implicit download or random-data fallback."""

from pathlib import Path
import numpy as np
import torch

from ..models.checkpoint import file_hash


class PackedTokenStream:
    def __init__(self, path, batch_size, seq_len, seed=42):
        self.tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.tokens.dtype.kind not in "iu" or self.tokens.ndim not in (1, 2):
            raise ValueError(
                "Expected a 1D token stream or 2D packed integer .npy array"
            )
        if (self.tokens.ndim == 1 and self.tokens.size < seq_len) or (
            self.tokens.ndim == 2 and self.tokens.shape[1] != seq_len
        ):
            raise ValueError("Token input incompatible with seq_len")
        if len(self.tokens) == 0 or batch_size < 1 or seq_len < 2:
            raise ValueError("Empty tokens or invalid batch/sequence size")
        self.batch_size, self.seq_len = batch_size, seq_len
        self.rng = np.random.default_rng(seed)
        self.sha256 = file_hash(path)

    def __call__(self):
        if self.tokens.ndim == 1:
            starts = self.rng.integers(
                0, len(self.tokens) - self.seq_len + 1, self.batch_size
            )
            batch = np.stack([self.tokens[i : i + self.seq_len] for i in starts])
        else:
            indices = self.rng.integers(0, len(self.tokens), self.batch_size)
            batch = np.asarray(self.tokens[indices])
        return torch.tensor(batch.astype(np.int64))

    def state_dict(self):
        return {"sha256": self.sha256, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if state["sha256"] != self.sha256:
            raise ValueError("Resume token input hash mismatch")
        self.rng.bit_generator.state = state["rng"]


class SyntheticTokenStream:
    """Explicitly a demo stream, with resumable local RNG."""

    def __init__(self, vocab_size, seq_len, batch_size, seed=42, enriched=False):
        self.vocab_size, self.seq_len, self.batch_size = vocab_size, seq_len, batch_size
        self.rng = np.random.default_rng(seed)
        self.enriched = enriched

    def __call__(self):
        x = self.rng.integers(0, self.vocab_size, (self.batch_size, self.seq_len))
        if self.enriched and self.seq_len >= 8:
            x[:, -3:-1] = x[:, 1:3]
        return torch.tensor(x, dtype=torch.long)

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
