from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from forecast_core.data_pipelines.omni_dscovr_sdo.feature_engineering import FEATURE_COLUMNS
from forecast_core.solar_encoder.embedding_loader import EmbeddingStore


@dataclass(slots=True)
class InferenceBatch:
    image_embed: torch.Tensor
    image_embed_mask: torch.Tensor
    history: torch.Tensor
    context: torch.Tensor
    persistence: torch.Tensor
    reference_timestamp: datetime


def _to_utc(ts: datetime | str | None) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def build_context_from_history(history_np: np.ndarray) -> np.ndarray:
    bz = history_np[:, 3]
    speed = history_np[:, 0]
    density = history_np[:, 4]
    latest_bz = float(bz[-1])
    latest_speed = float(speed[-1])
    latest_density = float(density[-1])
    return np.array([
        1.0 if latest_bz < 0 else 0.0,
        1.0 if latest_bz < -5.0 else 0.0,
        max(latest_density, 0.0) * max(latest_speed, 0.0) ** 2,
        float(bz[-3:].mean()),
        float(bz[-6:].mean()),
        float(speed[-6:].mean()),
        float(density[-6:].mean()),
    ], dtype=np.float32)


def _coerce_history(history_rows: list[dict[str, Any]]) -> np.ndarray:
    arr = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in history_rows], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f'history must have shape [T, {len(FEATURE_COLUMNS)}]')
    return arr


def build_inference_batch(
    payload: dict[str, Any],
    history_scaler,
    context_scaler,
    target_scaler,
    expected_embed_dim: int,
    embedding_store: EmbeddingStore | None = None,
    default_zero_embedding: bool = True,
    max_embedding_age_hours: float | None = 24.0,
) -> InferenceBatch:
    history_rows = payload.get('history') or []
    if not history_rows:
        raise ValueError('history is required for inference')

    history_np = _coerce_history(history_rows)
    context_np = build_context_from_history(history_np)
    persistence_np = history_np[-1, 1:].astype(np.float32)

    reference_timestamp = _to_utc(payload.get('reference_timestamp'))

    embedding = payload.get('surya_embedding')
    if embedding is not None:
        image_embed_np = np.asarray(embedding, dtype=np.float32)
        if image_embed_np.shape != (expected_embed_dim,):
            raise ValueError(f'surya_embedding must have shape [{expected_embed_dim}]')
        image_embed_mask_np = np.array([1.0], dtype=np.float32)
    elif embedding_store is not None:
        result = embedding_store.lookup(
            reference_timestamp,
            max_age_hours=max_embedding_age_hours,
            default_zero=default_zero_embedding,
        )
        image_embed_np = result.embedding
        image_embed_mask_np = np.array([1.0 if result.found else 0.0], dtype=np.float32)
    elif default_zero_embedding:
        image_embed_np = np.zeros(expected_embed_dim, dtype=np.float32)
        image_embed_mask_np = np.array([0.0], dtype=np.float32)
    else:
        raise ValueError('surya_embedding missing and no embedding store configured')

    history_t = torch.tensor(history_np[None, :, :], dtype=torch.float32)
    context_t = torch.tensor(context_np[None, :], dtype=torch.float32)
    persistence_t = torch.tensor(persistence_np[None, :], dtype=torch.float32)
    image_t = torch.tensor(image_embed_np[None, :], dtype=torch.float32)
    image_mask_t = torch.tensor(image_embed_mask_np[None, :], dtype=torch.float32)

    history_t = history_scaler.normalize(history_t)
    context_t = context_scaler.normalize(context_t)
    persistence_t = target_scaler.normalize(persistence_t)

    return InferenceBatch(
        image_embed=image_t,
        image_embed_mask=image_mask_t,
        history=history_t,
        context=context_t,
        persistence=persistence_t,
        reference_timestamp=reference_timestamp,
    )
