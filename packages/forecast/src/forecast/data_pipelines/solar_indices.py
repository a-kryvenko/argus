"""Build calibrated JB2008 solar-index estimates from GOES observations.

GOES measures irradiance and Mg II observables, not S10, M10, or Y10 in the
SET/JB2008 scale. Consequently this module keeps daily feature construction
separate from calibration against delayed authoritative SOLFSMY records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

GOES_IRRADIANCE_COLUMNS = (
    "goes_euv_256",
    "goes_euv_284",
    "goes_euv_304",
    "goes_euv_1175",
    "goes_lya_1216",
    "goes_euv_1335",
    "goes_euv_1405",
)

REQUIRED_GOES_COLUMNS = (
    "timestamp",
    "goes_au_factor",
    "goes_mgii_index",
    "goes_xray_background",
    "goes_euvs_quality_valid",
    *GOES_IRRADIANCE_COLUMNS,
)

INDEX_FEATURE_COLUMNS = {
    "s10": (
        "goes_euv_256_1au",
        "goes_euv_284_1au",
        "goes_euv_304_1au",
    ),
    "m10": ("goes_mgii_index",),
    "y10": (
        "goes_lya_1216_1au",
        "goes_lya_1216_1au_squared",
        "goes_xray_background",
        "goes_mgii_index",
        "goes_xray_background_x_mgii",
    ),
}


@dataclass(frozen=True)
class SolarIndexCalibration:
    """Serializable linear calibration evaluated on standardized features."""

    feature_columns: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing solar-index calibration features: {missing}")

        values = frame.loc[:, list(self.feature_columns)].to_numpy(dtype=float)
        means = np.asarray(self.feature_means, dtype=float)
        scales = np.asarray(self.feature_scales, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        predictions = self.intercept + ((values - means) / scales) @ coefficients

        if not np.isfinite(predictions).all():
            raise ValueError("Solar-index calibration produced non-finite values")
        return predictions


def _require_columns(frame: pd.DataFrame, columns, source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} frame is missing columns: {missing}")


def build_daily_goes_features(goes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid minute GOES records into calibration-ready daily rows."""
    _require_columns(goes, REQUIRED_GOES_COLUMNS, "GOES")
    if goes.empty:
        raise ValueError("Cannot build solar-index features from an empty GOES frame")

    records = goes.copy()
    records["timestamp"] = pd.to_datetime(
        records["timestamp"],
        utc=True,
        errors="coerce",
    )
    records = records.loc[records["timestamp"].notna()]
    records["solar_day"] = records["timestamp"].dt.floor("D")

    total_samples = records.groupby("solar_day").size().rename("goes_sample_count")
    records = records.loc[records["goes_euvs_quality_valid"].eq(True)].copy()
    if records.empty:
        raise ValueError("GOES frame contains no quality-valid EUVS observations")

    numeric_columns = [
        "goes_au_factor",
        "goes_mgii_index",
        "goes_xray_background",
        *GOES_IRRADIANCE_COLUMNS,
    ]
    records[numeric_columns] = records[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = np.isfinite(records[numeric_columns]).all(axis=1)
    records = records.loc[finite].copy()
    if records.empty:
        raise ValueError("GOES frame contains no finite complete observations")

    for column in GOES_IRRADIANCE_COLUMNS:
        records[f"{column}_1au"] = records[column] * records["goes_au_factor"]

    records["goes_lya_1216_1au_squared"] = records["goes_lya_1216_1au"] ** 2
    records["goes_xray_background_x_mgii"] = (
        records["goes_xray_background"] * records["goes_mgii_index"]
    )

    feature_columns = sorted({
        column
        for columns in INDEX_FEATURE_COLUMNS.values()
        for column in columns
    })
    daily = records.groupby("solar_day")[feature_columns].median()
    valid_samples = records.groupby("solar_day").size().rename("goes_valid_sample_count")
    daily = daily.join(total_samples).join(valid_samples)
    daily["goes_valid_fraction"] = (
        daily["goes_valid_sample_count"] / daily["goes_sample_count"]
    )

    daily = daily.reset_index().rename(columns={"solar_day": "timestamp"})
    daily["timestamp"] = daily["timestamp"] + pd.Timedelta(hours=12)
    return daily.sort_values("timestamp").reset_index(drop=True)


def _fit_calibration(
    frame: pd.DataFrame,
    target: str,
    feature_columns: tuple[str, ...],
) -> SolarIndexCalibration:
    complete = frame.dropna(subset=[target, *feature_columns])
    minimum_samples = len(feature_columns) + 1
    if len(complete) < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} overlapping records are required to "
            f"calibrate {target}; got {len(complete)}"
        )

    values = complete.loc[:, list(feature_columns)].to_numpy(dtype=float)
    targets = complete[target].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError(f"Non-finite values found while calibrating {target}")

    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    design = np.column_stack([np.ones(len(values)), (values - means) / scales])
    parameters, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(f"Calibration features for {target} are rank deficient")

    return SolarIndexCalibration(
        feature_columns=feature_columns,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        intercept=float(parameters[0]),
        coefficients=tuple(float(value) for value in parameters[1:]),
    )


def fit_solar_index_calibrations(
    goes: pd.DataFrame,
    solfsmy: pd.DataFrame,
) -> dict[str, SolarIndexCalibration]:
    """Fit GOES-to-SET mappings on their common UTC dates."""
    _require_columns(solfsmy, ("timestamp", *INDEX_FEATURE_COLUMNS), "SOLFSMY")

    features = build_daily_goes_features(goes)
    truth = solfsmy[["timestamp", *INDEX_FEATURE_COLUMNS]].copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"], utc=True, errors="coerce")
    truth[list(INDEX_FEATURE_COLUMNS)] = truth[list(INDEX_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    truth["timestamp"] = truth["timestamp"].dt.floor("D") + pd.Timedelta(hours=12)
    truth = truth.dropna(subset=["timestamp"]).drop_duplicates(
        "timestamp",
        keep="last",
    )
    overlap = features.merge(truth, on="timestamp", how="inner")
    if overlap.empty:
        raise ValueError("GOES and SOLFSMY frames have no overlapping UTC dates")

    return {
        index_name: _fit_calibration(overlap, index_name, feature_columns)
        for index_name, feature_columns in INDEX_FEATURE_COLUMNS.items()
    }


def extract_solar_indices(
    goes: pd.DataFrame,
    calibrations: Mapping[str, SolarIndexCalibration],
) -> pd.DataFrame:
    """Apply calibrated GOES-to-SET mappings and return daily index estimates."""
    missing = [name for name in INDEX_FEATURE_COLUMNS if name not in calibrations]
    if missing:
        raise ValueError(f"Missing solar-index calibrations: {missing}")

    result = build_daily_goes_features(goes)
    for index_name in INDEX_FEATURE_COLUMNS:
        result[index_name] = calibrations[index_name].predict(result)

    return result
