"""TabCausal graph head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import tabcausal_init_weights


class TabCausalGraphHead(nn.Module):
    """Directed edge scorer used by TabCausal.

    Each variable embedding is projected into source and target spaces.  The
    directed logit for ``i -> j`` is a temperature-scaled cosine similarity
    between the source projection of variable ``i`` and the target projection of
    variable ``j``.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        cosine_temp_init: float = 2.0,
        logit_bias_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.u_proj = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim_out))
        self.v_proj = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim_out))
        self.apply(tabcausal_init_weights)
        self.learned_temp = nn.Parameter(torch.tensor(float(cosine_temp_init)))
        self.final_bias = nn.Parameter(torch.tensor(float(logit_bias_init)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.u_proj(h), p=2, dim=-1)
        v = F.normalize(self.v_proj(h), p=2, dim=-1)
        logits = torch.matmul(u, v.transpose(-2, -1))
        return logits * torch.exp(self.learned_temp) + self.final_bias
