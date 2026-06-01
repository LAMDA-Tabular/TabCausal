"""TabCausal encoder.

This file contains only the inference architecture.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def tabcausal_init_weights(module: nn.Module) -> None:
    """Apply fan-in uniform initialization for linear layers."""

    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class AxialTransformerBlock(nn.Module):
    """One pre-normalized attention block followed by a feed-forward block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        key_size: int,
        widening_factor: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.key_size = int(key_size)
        self.inner_dim = self.num_heads * self.key_size

        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, self.inner_dim)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, widening_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(widening_factor * embed_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.apply(tabcausal_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_axis, seq_len, embed_dim = x.shape
        residual = x

        q = self.q_proj(self.ln_q(x)).view(batch_axis, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.k_proj(self.ln_k(x)).view(batch_axis, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.v_proj(self.ln_v(x)).view(batch_axis, seq_len, self.num_heads, self.key_size).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout1.p if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch_axis, seq_len, self.inner_dim)
        x = residual + self.dropout1(self.out_proj(attention))

        residual = x
        x = residual + self.dropout2(self.ffn(self.ln_ffn(x)))
        return x


# Keep this alias so older checkpoints whose state_dict keys contain
# ``feature_encoder.blocks.*`` load without renaming.
FlashTransformerBlock = AxialTransformerBlock


class TabCausalEncoder(nn.Module):
    """Alternating variable/sample axial encoder used by TabCausal."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        key_size: int,
        widening_factor: int,
        dropout: float,
        input_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        tabcausal_init_weights(self.input_proj)

        self.num_blocks = 2 * int(num_layers)
        self.blocks = nn.ModuleList(
            [
                AxialTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    key_size=key_size,
                    widening_factor=widening_factor,
                    dropout=dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode data shaped ``[batch, observations, variables, channels]``."""

        h = self.input_proj(x)
        for block in self.blocks:
            batch_size, num_obs, num_vars, embed_dim = h.shape
            h = h.reshape(batch_size * num_obs, num_vars, embed_dim)
            h = block(h)
            h = h.view(batch_size, num_obs, num_vars, embed_dim)
            h = h.permute(0, 2, 1, 3)

        h = self.final_ln(h)
        return h.max(dim=1).values
