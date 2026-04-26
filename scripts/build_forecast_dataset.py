#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    required = ["timestamp", "BX_GSM", "BY_GSM", "BZ_GSM", "V", "N", "T"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[required].sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def _merge_sources(omni: pd.DataFrame | None, dscovr: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    if omni is not None:
        frames.append(_normalize_columns(omni))
    if dscovr is not None:
        frames.append(_normalize_columns(dscovr))
    if not frames:
        raise ValueError("At least one of --omni or --dscovr must be provided")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp")
    
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.set_index("timestamp").resample("1h").mean().interpolate(limit=3).reset_index()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Argus forecast-ready time series CSV and optional NPZ dataset.")
    parser.add_argument("--omni", type=Path)
    parser.add_argument("--dscovr", type=Path)
    parser.add_argument("--out-csv", type=Path, default=Path("data/processed/forecast_timeseries.csv"))
    parser.add_argument("--build-npz", action="store_true")
    parser.add_argument("--npz-out", type=Path, default=Path("data/processed/forecast_dataset.npz"))
    parser.add_argument("--embeddings-path", type=Path)
    parser.add_argument("--history-hours", type=int, default=48)
    parser.add_argument("--forecast-hours", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=1280)
    parser.add_argument("--embedding-lag-hours", type=int, default=96)
    parser.add_argument("--max-embedding-age-hours", type=float, default=24.0)
    parser.add_argument("--require-embeddings", action="store_true")
    args = parser.parse_args()

    omni_df = _load_table(args.omni) if args.omni else None
    dscovr_df = _load_table(args.dscovr) if args.dscovr else None
    merged = _merge_sources(omni_df, dscovr_df)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"saved merged forecast time series to {args.out_csv} ({len(merged)} rows)")

    if not args.build_npz:
        return

    try:
        from forecast_core.data_pipelines.omni_dscovr_sdo.prepare_dataset import DatasetBuilder
        from forecast_core.solar_encoder.embedding_loader import EmbeddingStore
    except ImportError as exc:
        raise SystemExit(
            "Could not import DatasetBuilder from forecast_core. Activate your uv environment and install -e packages/forecast-core."
        ) from exc

    store = EmbeddingStore.from_npz(args.embeddings_path) if args.embeddings_path else None

    builder = DatasetBuilder(
        history_hours=args.history_hours,
        forecast_hours=args.forecast_hours,
        embed_dim=args.embed_dim,
        embedding_store=store,
        embedding_lag_hours=args.embedding_lag_hours,
        max_embedding_age_hours=args.max_embedding_age_hours,
        require_embeddings=args.require_embeddings,
    )
    dataset = builder.build(merged)
    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(args.npz_out, **dataset)
    print(f"saved forecast dataset to {args.npz_out}")


if __name__ == "__main__":
    main()
