from __future__ import annotations

import torch
import torch.nn as nn


class RegimeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_regimes: int = 4):
        super().__init__()
        self.num_regimes = num_regimes
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
