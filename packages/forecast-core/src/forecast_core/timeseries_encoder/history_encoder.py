from __future__ import annotations

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, T, F]
        x = history.transpose(1, 2)
        x = self.conv(x).mean(dim=-1)
        return self.proj(x)
