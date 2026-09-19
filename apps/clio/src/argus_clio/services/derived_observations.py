from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from clio.observations import wide_to_measurements as _wide_to_measurements
from clio.dataloaders.swpc_loader import SWPC_Loader
from clio.dataloaders.goes_loader import fetch_goes
from common.config import get_config
from argus_clio.services.calibration import (
    extract_solar_index_observations,
    load_solar_index_calibrations,
)
from requests import RequestException
REQUIRED_METRICS = SWPC_Loader.METRICS
SOLAR_INDEX_METRICS = ("s10", "m10", "y10")
OBSERVATION_METRICS = (*REQUIRED_METRICS, *SOLAR_INDEX_METRICS)
logger = logging.getLogger(__name__)


def normalize_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """Turn narrow source measurements into complete hourly observations."""

    required_columns = {"metric", "value", "observed_at"}
    missing_columns = required_columns.difference(measurements.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Measurement frame is missing columns: {missing}")
    if measurements.empty:
        return pd.DataFrame(columns=["observed_at", *OBSERVATION_METRICS])

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(OBSERVATION_METRICS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )

    wide = frame.pivot_table(
        index="observed_at",
        columns="metric",
        values="value",
        aggfunc="last",
    )
    for metric in OBSERVATION_METRICS:
        if metric not in wide:
            wide[metric] = np.nan

    wide = wide[list(OBSERVATION_METRICS)].sort_index()
    required = wide[list(REQUIRED_METRICS)].dropna(how="all").resample("1h").first()
    required = required.interpolate(method="time", limit_area="inside")
    required = required.fillna(required.mean(numeric_only=True))
    required = required.dropna(subset=list(REQUIRED_METRICS))
    # Calibrated estimates are optional: never interpolate, backfill, or
    # propagate them into hours without a GOES-derived observation.
    solar = wide[list(SOLAR_INDEX_METRICS)].resample("1h").last()
    wide = required.join(solar)
    wide.columns.name = None
    return wide.reset_index()


def load_solar_index_measurements(now: datetime) -> pd.DataFrame:
    """Use the configured frozen calibration; failures leave indices absent."""
    empty = pd.DataFrame(columns=["metric", "value", "observed_at"])
    config = get_config()
    registry = config.models_registry.get("models", {}).get("solar_index_calibration", {})
    path = registry.get("calibration_path")
    if not path:
        logger.warning("Solar-index observations unavailable: no calibration configured")
        return empty
    try:
        calibrations = load_solar_index_calibrations(config.workdir / Path(path))
        goes = fetch_goes(end=now)
        estimates = extract_solar_index_observations(goes, calibrations, pd.Timestamp(now))
    except FileNotFoundError as exc:
        logger.warning(
            "Solar-index calibration file not found: %s. S10/M10/Y10 remain "
            "unavailable; other observations continue updating. Train a calibration "
            "with forecast_core.data_pipelines.calibrate_solar_indices or set "
            "models.solar_index_calibration.calibration_path in models_registry.yaml "
            "to an existing artifact.",
            exc.filename,
        )
        return empty
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, RequestException):
        logger.exception("Solar-index observations unavailable: calibration or GOES failure")
        return empty

    # The live feed is a rolling day. Only persist the current hour snapshot,
    # so a later truncated feed cannot revise earlier daily estimates.
    estimates = estimates.loc[estimates["observed_at"] == pd.Timestamp(now).floor("h")]
    if estimates.empty:
        latest = pd.to_datetime(goes["timestamp"], utc=True).max() if not goes.empty else None
        valid = goes.loc[goes["goes_euvs_quality_valid"].eq(True)]
        last_valid = pd.to_datetime(valid["timestamp"], utc=True).max() if not valid.empty else None
        logger.warning(
            "Solar-index observations unavailable: no recent valid GOES samples; "
            "requested_hour=%s, latest_sample=%s, latest_quality_valid=%s, "
            "quality_valid_rows=%d/%d",
            pd.Timestamp(now).floor("h"), latest, last_valid, len(valid), len(goes),
        )
        return empty
    return _wide_to_measurements(estimates)


