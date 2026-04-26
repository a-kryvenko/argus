from __future__ import annotations

import pandas as pd


def embedding_timestamp_for_forecast(reference_timestamp, lag_hours: int = 96) -> pd.Timestamp:
    ts = pd.Timestamp(reference_timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts - pd.Timedelta(hours=lag_hours)
