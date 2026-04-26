from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EmbeddingLookupResult:
    vector: np.ndarray
    mask: float
    age_hours: float


class EmbeddingStore:
    def __init__(self, timestamps: np.ndarray, embeddings: np.ndarray):
        self.timestamps = pd.to_datetime(pd.Series(timestamps), utc=True)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(self.timestamps) != len(self.embeddings):
            raise ValueError("timestamps and embeddings length mismatch")
        self._ts_ns = self.timestamps.view("int64").to_numpy()

    @classmethod
    def from_npz(cls, path: str | Path) -> "EmbeddingStore":
        data = np.load(Path(path), allow_pickle=False)
        return cls(data["timestamps"], data["embeddings"])

    def lookup_at_or_before(self, ts, *, max_age_hours: float | None = None, fallback_dim: int | None = None) -> EmbeddingLookupResult:
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        idx = bisect_right(self._ts_ns.tolist(), ts.value) - 1
        if idx < 0:
            dim = fallback_dim if fallback_dim is not None else self.embeddings.shape[1]
            return EmbeddingLookupResult(np.zeros((dim,), dtype=np.float32), 0.0, np.inf)
        emb_ts = self.timestamps.iloc[idx]
        age_hours = (ts - emb_ts).total_seconds() / 3600.0
        if max_age_hours is not None and age_hours > max_age_hours:
            dim = fallback_dim if fallback_dim is not None else self.embeddings.shape[1]
            return EmbeddingLookupResult(np.zeros((dim,), dtype=np.float32), 0.0, age_hours)
        return EmbeddingLookupResult(self.embeddings[idx].astype(np.float32, copy=False), 1.0, age_hours)
