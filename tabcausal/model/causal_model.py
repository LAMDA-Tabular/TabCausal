"""TabCausal model definition for directed graph prediction."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .encoder import TabCausalEncoder
from .heads import TabCausalGraphHead


@dataclass(frozen=True)
class TabCausalConfig:
    """Constructor options for the public model."""

    embed_dim: int = 128
    out_dim: int | None = None
    col_num_blocks: int = 8
    col_nhead: int = 8
    key_size: int = 32
    widening_factor: int = 4
    dropout: float = 0.1
    cosine_temp_init: float = 2.0
    logit_bias_init: float = -1.0
    input_dim: int = 2

    @classmethod
    def from_dict(cls, values: dict) -> "TabCausalConfig":
        allowed = set(cls.__dataclass_fields__)
        clean = {key: values[key] for key in allowed if key in values}
        return cls(**clean)

    def to_dict(self) -> dict:
        return asdict(self)


class TabCausalModel(nn.Module):
    """TabCausal graph predictor."""

    def __init__(
        self,
        embed_dim: int = 128,
        out_dim: int | None = None,
        col_num_blocks: int = 8,
        col_nhead: int = 8,
        key_size: int = 32,
        widening_factor: int = 4,
        dropout: float = 0.1,
        cosine_temp_init: float = 2.0,
        logit_bias_init: float = -1.0,
        input_dim: int = 2,
    ) -> None:
        super().__init__()
        out_dim = out_dim or embed_dim
        self.config = TabCausalConfig(
            embed_dim=embed_dim,
            out_dim=out_dim,
            col_num_blocks=col_num_blocks,
            col_nhead=col_nhead,
            key_size=key_size,
            widening_factor=widening_factor,
            dropout=dropout,
            cosine_temp_init=cosine_temp_init,
            logit_bias_init=logit_bias_init,
            input_dim=input_dim,
        )
        self.feature_encoder = TabCausalEncoder(
            embed_dim=embed_dim,
            num_layers=col_num_blocks,
            num_heads=col_nhead,
            key_size=key_size,
            widening_factor=widening_factor,
            dropout=dropout,
            input_dim=input_dim,
        )
        self.graph_head = TabCausalGraphHead(
            dim_in=embed_dim,
            dim_out=out_dim,
            cosine_temp_init=cosine_temp_init,
            logit_bias_init=logit_bias_init,
        )

    def extract_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.extract_feature_embeddings(x)
        return self.graph_head(embeddings)
