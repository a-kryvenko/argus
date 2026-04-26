from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EmbeddingLookupResult:
    embedding: np.ndarray
    found: bool
    source_timestamp: pd.Timestamp | None
    age_hours: float | None


class EmbeddingStore:
    """Timestamp-keyed access to precomputed Surya embeddings.

    Canonical contract:
    - timestamps: array-like of ISO timestamps or datetime-like values
    - embeddings: float32 array with shape [N, D]

    Lookup behavior is operationally useful by default: the store returns the
    nearest embedding at or before the requested timestamp, optionally bounded
    by a maximum age.
    """

    def __init__(self, timestamps: pd.DatetimeIndex, embeddings: np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError('embeddings must have shape [N, D]')
        if len(timestamps) != embeddings.shape[0]:
            raise ValueError('timestamps and embeddings length mismatch')

        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).sort_values()
        order = np.argsort(ts.view('i8'))
        self.timestamps = pd.DatetimeIndex(ts[order])
        self.embeddings = np.asarray(embeddings, dtype=np.float32)[order]
        self.embed_dim = self.embeddings.shape[-1]

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        timestamp_key: str = 'timestamps',
        embed_key: str = 'embeddings',
    ) -> 'EmbeddingStore':
        payload = np.load(path, allow_pickle=True)
        timestamps = payload[timestamp_key]
        embeddings = payload[embed_key].astype(np.float32)
        return cls(pd.to_datetime(timestamps, utc=True), embeddings)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        timestamp_col: str = 'timestamp',
    ) -> 'EmbeddingStore':
        df = pd.read_csv(path)
        timestamps = pd.to_datetime(df[timestamp_col], utc=True)
        feature_cols = [c for c in df.columns if c != timestamp_col]
        embeddings = df[feature_cols].to_numpy(dtype=np.float32)
        return cls(timestamps, embeddings)

    @staticmethod
    def _normalize_timestamp(timestamp: Any) -> pd.Timestamp:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')

    def lookup(
        self,
        timestamp: Any,
        *,
        max_age_hours: float | None = None,
        default_zero: bool = False,
    ) -> EmbeddingLookupResult:
        target_ts = self._normalize_timestamp(timestamp)
        idx = self.timestamps.searchsorted(target_ts, side='right') - 1
        if idx < 0:
            if default_zero:
                return EmbeddingLookupResult(
                    embedding=np.zeros(self.embed_dim, dtype=np.float32),
                    found=False,
                    source_timestamp=None,
                    age_hours=None,
                )
            raise KeyError(f'No embedding at or before timestamp {target_ts!s}')

        source_ts = self.timestamps[idx]
        age_hours = float((target_ts - source_ts) / pd.Timedelta(hours=1))
        if max_age_hours is not None and age_hours > max_age_hours:
            if default_zero:
                return EmbeddingLookupResult(
                    embedding=np.zeros(self.embed_dim, dtype=np.float32),
                    found=False,
                    source_timestamp=source_ts,
                    age_hours=age_hours,
                )
            raise KeyError(
                f'Nearest embedding for {target_ts!s} is too old: '
                f'{age_hours:.2f}h > {max_age_hours:.2f}h'
            )

        return EmbeddingLookupResult(
            embedding=self.embeddings[idx],
            found=True,
            source_timestamp=source_ts,
            age_hours=age_hours,
        )

    def get(
        self,
        timestamp: Any,
        default_zero: bool = False,
        max_age_hours: float | None = None,
    ) -> np.ndarray:
        return self.lookup(timestamp, default_zero=default_zero, max_age_hours=max_age_hours).embedding
