from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from surya_adapter.common.pooling import pool_tokens


@dataclass(slots=True)
class BackboneConfig:
    embedding_dim: int
    token_pooling: str = "mean"
    normalize_embedding: bool = True


class SuryaBackboneAdapter(nn.Module):
    """Thin adapter over a vendor Surya-style model.

    Supported vendor behaviors:
    1. model(batch, return_embedding=True) -> [B, D]
    2. model(batch) -> [B, L, D] token tensor
    3. custom token extractor method name via token_method
    """

    def __init__(
        self,
        vendor_model: nn.Module,
        embedding_dim: int,
        token_pooling: str = "mean",
        normalize_embedding: bool = True,
        token_method: str | None = None,
    ) -> None:
        super().__init__()
        self.vendor_model = vendor_model
        self.embedding_dim = embedding_dim
        self.token_pooling = token_pooling
        self.normalize_embedding = normalize_embedding
        self.token_method = token_method

    def _normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        if not self.normalize_embedding:
            return embedding
        return embedding / (embedding.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def _call_vendor(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.token_method and hasattr(self.vendor_model, self.token_method):
            fn = getattr(self.vendor_model, self.token_method)
            return fn(batch)

        try:
            return self.vendor_model(batch, return_embedding=True)
        except TypeError:
            return self.vendor_model(batch)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._call_vendor(batch)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"vendor model returned {type(out)!r}, expected tensor")

        if out.ndim == 2:
            embedding = out
        elif out.ndim == 3:
            embedding = pool_tokens(out, strategy=self.token_pooling)
        else:
            raise ValueError(f"unsupported vendor output shape: {tuple(out.shape)}")

        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: expected {self.embedding_dim}, got {embedding.shape[-1]}"
            )
        return self._normalize(embedding)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(batch)
