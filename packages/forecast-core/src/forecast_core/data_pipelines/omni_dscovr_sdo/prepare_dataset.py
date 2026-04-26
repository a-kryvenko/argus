from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from forecast_core.data_pipelines.omni_dscovr_sdo.feature_engineering import FEATURE_COLUMNS, add_context_features, build_context_vector
from forecast_core.data_pipelines.omni_dscovr_sdo.joins import load_timeseries_csv
from forecast_core.data_pipelines.omni_dscovr_sdo.labels import build_bz_event_targets, build_targets
from forecast_core.solar_encoder.embedding_loader import EmbeddingStore, EmbeddingLookupResult


class DatasetBuilder:
    def __init__(
        self,
        history_hours: int = 48,
        forecast_hours: int = 96,
        embed_dim: int = 1280,
        bz_event_threshold: float = -5.0,
        embedding_store: EmbeddingStore | None = None,
        embedding_lag_hours: int = 96,
        max_embedding_age_hours: float | None = 24.0,
        require_embeddings: bool = False,
    ):
        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.embed_dim = embed_dim
        self.bz_event_threshold = bz_event_threshold

        self.embedding_store = embedding_store
        self.embedding_lag_hours = embedding_lag_hours
        self.max_embedding_age_hours = max_embedding_age_hours
        self.require_embeddings = require_embeddings

        self.history_columns = ["BX_GSM", "BY_GSM", "BZ_GSM", "V", "N", "T"]
        self.target_columns = ["BX_GSM", "BY_GSM", "BZ_GSM", "N"]

    def _normalize_timestamp_series(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values("timestamp").reset_index(drop=True)
        return out

    def _build_context(self, history_window: pd.DataFrame) -> np.ndarray:
        bz = history_window["BZ_GSM"].to_numpy(dtype=np.float32)
        v = history_window["V"].to_numpy(dtype=np.float32)
        n = history_window["N"].to_numpy(dtype=np.float32)

        context = np.array(
            [
                float(history_window["BX_GSM"].iloc[-1]),
                float(history_window["BY_GSM"].iloc[-1]),
                float(history_window["BZ_GSM"].iloc[-1]),
                float(history_window["V"].iloc[-1]),
                float(history_window["N"].iloc[-1]),
                float(history_window["T"].iloc[-1]),
                float(bz.min()),
                float((v * n).mean()),
            ],
            dtype=np.float32,
        )
        return context

    def _lookup_embedding(self, forecast_reference_ts: pd.Timestamp) -> EmbeddingLookupResult:
        if self.embedding_store is None:
            return EmbeddingLookupResult(
                vector=np.zeros((self.embed_dim,), dtype=np.float32),
                mask=0.0,
                age_hours=np.inf,
            )

        embedding_ts = forecast_reference_ts - pd.Timedelta(hours=self.embedding_lag_hours)

        return self.embedding_store.lookup_at_or_before(
            embedding_ts,
            max_age_hours=self.max_embedding_age_hours,
            fallback_dim=self.embed_dim,
        )

    def build(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        df = add_context_features(df)
        df = df.sort_values('timestamp').reset_index(drop=True)
        timestamps = pd.to_datetime(df['timestamp'], utc=True).tolist()

        image_embed = []
        image_embed_mask = []
        embedding_age_hours = []
        history = []
        context = []
        persistence = []
        targets = []
        bz_events = []
        sample_timestamps = []

        start = self.history_hours - 1
        end = len(df) - self.forecast_hours - 1
        for idx in range(start, end + 1):
            hist_rows = df.iloc[idx - self.history_hours + 1:idx + 1]
            if len(hist_rows) != self.history_hours:
                continue
            target_rows = df.iloc[idx + 1:idx + 1 + self.forecast_hours]
            if len(target_rows) != self.forecast_hours:
                continue
            ts = pd.Timestamp(timestamps[idx])
            embed, embed_mask, age_hours = self._lookup_embedding(ts)
            if self.require_embeddings and embed_mask.item() != 1.0:
                continue
            image_embed.append(embed)
            image_embed_mask.append(embed_mask)
            embedding_age_hours.append(age_hours)
            history.append(hist_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            context.append(build_context_vector(df, idx))
            persistence.append(hist_rows.iloc[-1][['BX_GSM', 'BY_GSM', 'BZ_GSM', 'N']].to_numpy(dtype=np.float32))
            targets.append(build_targets(df, idx, self.forecast_hours))
            bz_events.append(build_bz_event_targets(df, idx, self.forecast_hours))
            sample_timestamps.append(ts.isoformat())

        return {
            'timestamps': np.asarray(sample_timestamps, dtype='U32'),
            'image_embed': np.asarray(image_embed, dtype=np.float32),
            'image_embed_mask': np.asarray(image_embed_mask, dtype=np.float32),
            'embedding_age_hours': np.asarray(embedding_age_hours, dtype=np.float32),
            'history': np.asarray(history, dtype=np.float32),
            'context': np.asarray(context, dtype=np.float32),
            'persistence': np.asarray(persistence, dtype=np.float32),
            'targets': np.asarray(targets, dtype=np.float32),
            'bz_event_targets': np.asarray(bz_events, dtype=np.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--history-hours', type=int, default=48)
    parser.add_argument('--forecast-hours', type=int, default=96)
    parser.add_argument('--embed-dim', type=int, default=1280)
    parser.add_argument('--embeddings')
    parser.add_argument('--embedding-format', choices=['npz', 'csv'], default='npz')
    parser.add_argument('--max-embedding-age-hours', type=float, default=24.0)
    parser.add_argument('--require-embeddings', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_timeseries_csv(args.input)
    store = None
    if args.embeddings:
        store = EmbeddingStore.from_npz(args.embeddings) if args.embedding_format == 'npz' else EmbeddingStore.from_csv(args.embeddings)
    builder = DatasetBuilder(
        args.history_hours,
        args.forecast_hours,
        args.embed_dim,
        store,
        max_embedding_age_hours=args.max_embedding_age_hours,
        require_embeddings=args.require_embeddings,
    )
    dataset = builder.build(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    print(f'saved {args.output}')


if __name__ == '__main__':
    # Should never be called from command line
    # main()
    pass
