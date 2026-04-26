from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_jsoc_request_manifest(timestamps: pd.Series, out_path: str | Path) -> Path:
    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True)})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
