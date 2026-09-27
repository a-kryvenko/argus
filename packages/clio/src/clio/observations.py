"""Retrieve source observations; no model calibration or gap filling."""
import pandas as pd
from clio.dataloaders.swpc_loader import SWPC_Loader

REQUIRED_METRICS = SWPC_Loader.METRICS
SOLAR_INDEX_METRICS = ("s10", "m10", "y10")
OBSERVATION_METRICS = (*REQUIRED_METRICS, *SOLAR_INDEX_METRICS)
# Rotation DLinear needs 57 days of context plus a bounded fill buffer.
HISTORY_DAYS = 60
LIVE_SOURCE_DAYS = 6

load_live_measurements = SWPC_Loader.load_measurements

def wide_to_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = "issue_time" if "issue_time" in frame else "observed_at"
    metrics = [metric for metric in OBSERVATION_METRICS if metric in frame]
    if timestamp_column not in frame or not metrics:
        return pd.DataFrame(columns=["metric", "value", "observed_at"])

    return (
        frame
        .melt(
            id_vars=timestamp_column,
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
        .rename(columns={timestamp_column: "observed_at"})
        .dropna(subset=["observed_at", "value"])
    )
