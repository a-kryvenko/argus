"""Clio-owned observation bootstrap cache for JB2008 solar backgrounds."""
import hashlib
import logging
from pathlib import Path
import tempfile

import pandas as pd
import requests
from clio.dataloaders.goes_history_loader import (
    archive_session, discover_goes_history, iter_goes_history,
)
from clio.dataloaders.gfz_loader import load_gfz_f107
from common.config import get_config
from argus_clio.services.observations.calibration import extract_solar_indices, load_solar_index_calibrations

logger = logging.getLogger(__name__)
COLUMNS = ["metric", "value", "observed_at"]



def merge_history(history: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Prefer application measurements for each solar metric/day as a group."""
    history, observations = history.copy(), observations.copy()
    for frame in (history, observations):
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        frame["_day"] = frame.observed_at.dt.floor("D")
    keys = pd.MultiIndex.from_frame(observations[["metric", "_day"]])
    history = history.loc[~pd.MultiIndex.from_frame(history[["metric", "_day"]]).isin(keys)]
    return pd.concat([history[COLUMNS], observations[COLUMNS]], ignore_index=True)


def download_solar_index_history(start, end, config) -> pd.DataFrame:
    # Both providers supply observations. No input-forecast files are consulted.
    logger.info("JB2008: downloading observed F10.7 history from GFZ")
    flux = load_gfz_f107(start, end)
    registry = config.models_registry["models"]["solar_index_calibration"]
    calibration = load_solar_index_calibrations(config.workdir / registry["calibration_path"])
    archive = registry.get("goes_archive", {})
    satellites = archive.get("satellites", [18, 16])
    cache_dir = config.workdir / "data/observations/jb2008/goes_archive"
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    frames = [flux]
    with archive_session() as session:
        logger.info("JB2008: discovering GOES archives for %s", satellites)
        files = discover_goes_history(sorted(set(days.year)), satellites, session)
        for _, goes in iter_goes_history(files, days, satellites, cache_dir, session, refresh=True):
            if goes.empty:
                continue
            # Restrict to completed days and observations before issuance.
            goes = goes.loc[goes.timestamp.lt(end.floor("D"))]
            if goes.empty:
                continue
            logger.info("JB2008: calibrating %d archived GOES samples", len(goes))
            daily = extract_solar_indices(goes, calibration)
            frames.append(daily.melt(id_vars="timestamp", value_vars=["s10", "m10", "y10"],
                                     var_name="metric", value_name="value")
                          .rename(columns={"timestamp": "observed_at"}))
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def load_density_history(issue_time) -> pd.DataFrame:
    """Refresh at most daily, preserving partial coverage for later observations.

    A calibration fingerprint prevents mixing estimates from different models.
    This function is invoked only after DB history proves insufficient.
    """
    config = get_config()
    entry = config.models_registry["models"].get("atmospheric_density", {})
    cache_dir = config.workdir / entry.get("history_cache", "data/observations/jb2008")
    calibration_path = config.workdir / config.models_registry["models"]["solar_index_calibration"]["calibration_path"]
    fingerprint = hashlib.sha256(calibration_path.read_bytes()).hexdigest()[:16]
    path = cache_dir / f"solar-{fingerprint}.parquet"
    issue = pd.Timestamp(issue_time)
    start = issue - pd.Timedelta(days=88)
    cached = pd.DataFrame(columns=COLUMNS)
    if path.is_file():
        logger.info("JB2008: reading history cache %s", path.name)
        cached = pd.read_parquet(path)
        fetched = cached.attrs.get("fetched_at")
        if fetched is not None and pd.Timestamp(fetched).floor("D") == issue.floor("D"):
            return cached.loc[cached.observed_at.between(start, issue)]
    try:
        downloaded = download_solar_index_history(start, issue, config)
    except (requests.RequestException, OSError, ValueError, KeyError, RuntimeError):
        logger.exception("JB2008 history refresh failed; retaining cached observations")
        return cached.loc[cached.observed_at.between(start, issue)]
    merged = pd.concat([cached, downloaded], ignore_index=True).drop_duplicates(
        ["metric", "observed_at"], keep="last")
    merged = merged.loc[merged.observed_at.between(start, issue)].sort_values("observed_at")
    merged.attrs["fetched_at"] = issue.isoformat()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".parquet", delete=False) as temporary:
        tmp_path = Path(temporary.name)
    try:
        merged.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return merged
