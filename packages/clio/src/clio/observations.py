"""Retrieve source observations; no model calibration or gap filling."""
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from common.config import get_config
from clio.dataloaders.spdf_loader import SPDF_Loader
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


def load_bootstrap_measurements(now: datetime) -> pd.DataFrame:
    config = get_config()
    legacy_path = config.project_config.get("paths", {}).get("live_sensors")
    if legacy_path:
        legacy_path = config.workdir / Path(legacy_path)
        if legacy_path.is_file():
            legacy = pd.read_csv(legacy_path, parse_dates=["issue_time"])
            measurements = wide_to_measurements(legacy)
            if not measurements.empty:
                return measurements

    historical = SPDF_Loader.load(
        start_date=now - timedelta(days=HISTORY_DAYS),
        end_date=now - timedelta(days=LIVE_SOURCE_DAYS - 1),
    )
    return wide_to_measurements(historical)

