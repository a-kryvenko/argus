"""Calibrated GOES historical solar indices."""
import logging
import pandas as pd
from clio.providers.goes_history_loader import archive_session, discover_goes_history, iter_goes_history
from clio.observations.calibration import extract_solar_indices, load_solar_index_calibrations
logger = logging.getLogger(__name__)
COLUMNS = ['metric', 'value', 'observed_at']


def download_goes_index_history(start, end, config) -> pd.DataFrame:
    """Calibrated GOES archive observations, independent of GFZ availability."""
    registry = config.models_registry["models"]["solar_index_calibration"]
    calibration = load_solar_index_calibrations(config.workdir / registry["calibration_path"])
    archive = registry.get("goes_archive", {})
    satellites = archive.get("satellites", [18, 16])
    cache_dir = config.workdir / "data/observations/jb2008/goes_archive"
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    frames = []
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
    return pd.concat(frames, ignore_index=True)[COLUMNS] if frames else pd.DataFrame(columns=COLUMNS)

