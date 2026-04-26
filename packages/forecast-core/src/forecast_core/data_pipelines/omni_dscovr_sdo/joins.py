from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_timeseries_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
