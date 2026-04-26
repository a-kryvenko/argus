from __future__ import annotations

import torch
import torch.nn as nn


class ProbabilisticRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.log_std_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, persistence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mean = self.mean_head(x)
        gate = self.residual_gate(x)
        mean = gate * persistence + (1.0 - gate) * raw_mean
        log_std = torch.clamp(self.log_std_head(x), min=-5.0, max=2.0)
        std = torch.exp(log_std)
        return mean, std
