"""Token embedding with learnable absolute positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Token embedding + learnable absolute positional encoding.

    Absolute positional encoding is important for the recurrent mixers
    (Gated DeltaNet) because they lack an inherent absolute-position mechanism;
    the embedding supplies it explicitly.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Match the original GPT-2 style init for the controlled models.
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed token ids and add positional encoding.

        Args:
            input_ids: (B, L) long tensor of token indices.

        Returns:
            (B, L, D) embedded representations.
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        x = self.token_emb(input_ids) * math.sqrt(self.d_model) + self.pos_emb(
            positions
        )
        return self.dropout(self.norm(x))
