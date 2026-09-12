"""Output heads for classification and sequence generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TaskHead(nn.Module):
    """Flexible output head.

    - ``classification``: last non-pad hidden state -> linear -> class logits.
    - ``generation``: every position -> vocab logits (next-token prediction).
    """

    def __init__(
        self,
        d_model: int = 512,
        output_type: str = "generation",
        num_classes: int = 64,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.output_type = output_type
        self.norm = nn.LayerNorm(d_model)

        if output_type == "classification":
            self.proj = nn.Linear(d_model, num_classes)
        elif output_type == "generation":
            assert vocab_size is not None
            self.proj = nn.Linear(d_model, vocab_size)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def forward(
        self,
        hidden: Tensor,
        labels: Tensor | None = None,
        input_ids: Tensor | None = None,
    ) -> dict:
        """Project hidden states to logits (and optionally compute CE loss).

        Args:
            hidden: (B, L, D) output of the last model layer.
            labels: (B,) for classification or (B, L) for generation.
            input_ids: (B, L) original tokens (for last-non-pad selection).

        Returns:
            dict with "logits" and optionally "loss".
        """
        hidden = self.norm(hidden)

        if self.output_type == "classification":
            if input_ids is not None:
                non_pad = input_ids != 0
                lengths = non_pad.sum(dim=-1).clamp(min=1) - 1
                last_hidden = hidden[
                    torch.arange(hidden.size(0), device=hidden.device), lengths
                ]
            else:
                last_hidden = hidden[:, -1, :]
            logits = self.proj(last_hidden)
            result = {"logits": logits}
            if labels is not None:
                result["loss"] = F.cross_entropy(logits, labels)

        elif self.output_type == "generation":
            logits = self.proj(hidden)
            result = {"logits": logits}
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                result["loss"] = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        return result
