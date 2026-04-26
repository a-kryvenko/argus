from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(slots=True)
class SuryaAdapterConfig:
    embed_dim: int = 1280
    projection_dim: int = 256
    dropout: float = 0.1


class SuryaAdapter(nn.Module):
    """Projection and normalization layer for precomputed Surya embeddings."""

    def __init__(self, config: SuryaAdapterConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_dim, config.projection_dim),
        )
        self.missing_embedding = nn.Parameter(torch.zeros(config.projection_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        projected = self.proj(x)
        if mask is None:
            return projected
        mask = mask.to(projected.dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        return projected * mask + self.missing_embedding.unsqueeze(0) * (1.0 - mask)
