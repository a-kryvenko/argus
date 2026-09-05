"""Read solar-calibration inputs on the UTC observation dates of training shards."""
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from forecast.data_pipelines.solar_indices import INDEX_FEATURE_COLUMNS, REQUIRED_GOES_COLUMNS


def training_observation_days(path: str | Path) -> pd.DatetimeIndex:
    """Scan only issue_time, deduplicating forecast horizons without loading features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    dataset = ds.dataset(path, format="parquet")
    if "issue_time" not in dataset.schema.names:
        raise ValueError(f"Training dataset has no issue_time column: {path}")
    days: set[pd.Timestamp] = set()
    for batch in dataset.scanner(columns=["issue_time"]).to_batches():
        timestamps = pd.to_datetime(batch.column(0).to_pandas(), utc=True, errors="coerce")
        days.update(timestamps.dropna().dt.floor("D").unique())
    if not days:
        raise ValueError(f"Training dataset has no valid observation dates: {path}")
    return pd.DatetimeIndex(sorted(days))


def select_observation_days(frame: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Restrict source records to UTC issue dates, never future valid_time dates."""
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    return result.loc[result["timestamp"].dt.floor("D").isin(days)].sort_values(
        "timestamp",
    ).reset_index(drop=True)


def read_solar_source(path: str | Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Solar calibration input not found: {path}")
    result = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    missing = set(required_columns).difference(result.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return result


def load_calibration_dataset(workdir: Path, entry: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized GOES history and SOLFSMY for one registry dataset entry."""
    days = training_observation_days(workdir / entry["training_path"])
    goes = read_solar_source(workdir / entry["goes_path"], REQUIRED_GOES_COLUMNS)
    truth = read_solar_source(
        workdir / entry["solfsmy_path"], ("timestamp", *INDEX_FEATURE_COLUMNS),
    )
    goes = select_observation_days(goes, days)
    truth = select_observation_days(truth, days)
    if goes.empty or truth.empty:
        raise ValueError("GOES and SOLFSMY must contain records on this split's observation dates")
    return goes, truth
