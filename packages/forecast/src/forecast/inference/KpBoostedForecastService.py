from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from common.adapters import kp_forecast_from_dataframe, observations_to_dataframe
from common.schemas.observation import Observation
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, hilbert

LEAD_HOURS_HORIZON = 72
HHT_VARIABLES = ("bz", "bt", "v", "n", "dynamic_pressure")
HHT_WINDOW_HOURS = 168
HHT_MAX_IMFS = 3
HHT_MIN_VALID = 48

DEFAULT_HORIZON_BUCKETS = {
    "0_3h": (0, 3),
    "3_6h": (3, 6),
    "6_12h": (6, 12),
    "12_24h": (12, 24),
    "24_48h": (24, 48),
    "48_72h": (48, 72),
}


class KpBoostedForecastService:
    """Inference service for the ordinal Kp bundle trained in notebook 9."""

    # The boosted service is a drop-in replacement for KpForecastService and uses
    # the same forecast product/registry entry.
    registry_name: str = "kp"

    def __init__(self, models: dict):
        self.model_bundle = self._resolve_model_bundle(models)

    @staticmethod
    def _resolve_model_bundle(models: dict) -> dict:
        if "models" in models and "ordinal_cuts" in models:
            return models

        for name in ("boosted", "ordinal"):
            candidate = models.get(name)
            if (
                isinstance(candidate, dict)
                and "models" in candidate
                and "ordinal_cuts" in candidate
            ):
                return candidate

        if len(models) == 1:
            candidate = next(iter(models.values()))
            if (
                isinstance(candidate, dict)
                and "models" in candidate
                and "ordinal_cuts" in candidate
            ):
                return candidate

        raise ValueError(
            "Kp boosted model bundle was not found; expected the bundle itself "
            "or a 'boosted'/'ordinal' registry model"
        )

    @staticmethod
    def forecast_from_df(df: pd.DataFrame):
        return kp_forecast_from_dataframe(df)

    def forecast(self, observations: Observation):
        issue_time = datetime.now(tz=timezone.utc)
        frame = self._prepare_frame(observations, issue_time)
        frame = self._forecast_kp(frame)
        return self.forecast_from_df(frame)

    def _prepare_frame(
        self, observations: Observation, issue_time: datetime
    ) -> pd.DataFrame:
        history = observations_to_dataframe(observations)
        history = self._add_issue_time_features(history)

        forecast_start_time = issue_time - timedelta(
            minutes=issue_time.minute,
            seconds=issue_time.second,
            microseconds=issue_time.microsecond,
        )
        horizon = self._forecast_horizon()
        last_row = history.iloc[[-1]].copy()
        frame = pd.concat([last_row] * horizon, ignore_index=True)
        frame["issue_time"] = issue_time
        frame["lead_hours"] = np.arange(1, horizon + 1)
        frame["valid_time"] = forecast_start_time + pd.to_timedelta(
            frame["lead_hours"], unit="h"
        )
        frame["lead_norm"] = frame["lead_hours"] / horizon
        frame["horizon_bucket"] = self._horizon_bucket(frame["lead_hours"])

        for name, value in self._latest_hht_features(history).items():
            frame[name] = value

        return frame

    def _forecast_horizon(self) -> int:
        buckets = self.model_bundle.get("horizon_buckets", DEFAULT_HORIZON_BUCKETS)
        if not buckets:
            return LEAD_HOURS_HORIZON
        return int(max(upper for _, upper in buckets.values()))

    def _horizon_bucket(self, lead_hours: pd.Series) -> pd.Series:
        buckets = self.model_bundle.get("horizon_buckets", DEFAULT_HORIZON_BUCKETS)
        result = pd.Series(pd.NA, index=lead_hours.index, dtype="object")
        for name, (lower, upper) in buckets.items():
            mask = lead_hours.gt(lower) & lead_hours.le(upper)
            result.loc[mask] = name

        if result.isna().any():
            invalid = sorted(lead_hours.loc[result.isna()].unique().tolist())
            raise ValueError(f"Unassigned lead hours: {invalid}")
        return result

    @staticmethod
    def _add_issue_time_features(history: pd.DataFrame) -> pd.DataFrame:
        if len(history) <= 6:
            raise ValueError("At least 7 hourly observations are required")

        frame = history.copy()
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
        frame = frame.sort_values("issue_time").drop_duplicates("issue_time", keep="last")
        frame = frame.set_index("issue_time")

        for hours in (3, 6):
            frame[f"kp_mean_{hours}h"] = frame["kp"].rolling(
                f"{hours}h", min_periods=1
            ).mean()
            frame[f"kp_delta_{hours}h"] = frame["kp"] - frame["kp"].shift(hours)

        frame["bz_min_1h"] = frame["bz"].rolling("1h", min_periods=1).min()
        frame["bz_min_3h"] = frame["bz"].rolling("3h", min_periods=1).min()
        frame["bz_mean_3h"] = frame["bz"].rolling("3h", min_periods=1).mean()
        frame["bz_delta_3h"] = frame["bz"] - frame["bz"].shift(3)

        # Match the training table: Bt is the transverse GSM field magnitude,
        # while dynamic_pressure is the unscaled n * V^2 proxy.
        frame["bt"] = np.sqrt(frame["by"].pow(2) + frame["bz"].pow(2))
        frame["southward_bz"] = (-frame["bz"]).clip(lower=0)
        frame["dynamic_pressure"] = frame["n"] * frame["v"].pow(2)
        frame["v_x_southward_bz"] = frame["v"] * frame["southward_bz"]

        clock_angle = np.mod(np.arctan2(frame["by"], frame["bz"]), 2 * np.pi)
        frame["newell_coupling"] = (
            frame["v"].clip(lower=0).pow(4 / 3)
            * frame["bt"].clip(lower=0).pow(2 / 3)
            * np.abs(np.sin(clock_angle / 2)).pow(8 / 3)
        )
        for hours in (1, 3, 6):
            frame[f"newell_integral_{hours}h"] = frame["newell_coupling"].rolling(
                f"{hours}h", min_periods=1
            ).sum()

        return frame.reset_index()

    def _latest_hht_features(self, history: pd.DataFrame) -> dict[str, float]:
        required = set(self.model_bundle.get("hht_features", ()))
        required.update(
            feature
            for features in self.model_bundle.get(
                "feature_columns_by_bucket", {}
            ).values()
            for feature in features
            if feature.startswith("hht_")
        )
        if not required:
            return {}

        window = history.tail(HHT_WINDOW_HOURS)
        features: dict[str, float] = {}
        for variable in HHT_VARIABLES:
            features.update(self._hht_window_summary(window[variable].to_numpy(), variable))
        return {name: features.get(name, np.nan) for name in required}

    @classmethod
    def _hht_window_summary(
        cls, values: np.ndarray, prefix: str
    ) -> dict[str, float]:
        imfs, residue = cls._empirical_mode_decomposition(values)
        result: dict[str, float] = {}
        for index in range(HHT_MAX_IMFS):
            stem = f"hht_{prefix}_imf{index + 1}"
            if index >= len(imfs):
                result.update(
                    {
                        f"{stem}_value": np.nan,
                        f"{stem}_amplitude": np.nan,
                        f"{stem}_frequency": np.nan,
                        f"{stem}_energy": np.nan,
                    }
                )
                continue

            imf = imfs[index]
            analytic = hilbert(imf)
            amplitude = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            frequency = np.r_[np.nan, np.diff(phase) / (2 * np.pi)]
            result.update(
                {
                    f"{stem}_value": float(imf[-1]),
                    f"{stem}_amplitude": float(amplitude[-1]),
                    f"{stem}_frequency": float(frequency[-1]),
                    f"{stem}_energy": float(np.mean(imf**2)),
                }
            )
        result[f"hht_{prefix}_residue"] = (
            float(residue[-1]) if np.isfinite(residue[-1]) else np.nan
        )
        return result

    @staticmethod
    def _empirical_mode_decomposition(
        values: np.ndarray,
        max_imfs: int = HHT_MAX_IMFS,
        max_siftings: int = 10,
        stop_sd: float = 0.2,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        data = np.asarray(values, dtype=float)
        valid = np.isfinite(data)
        if valid.sum() < HHT_MIN_VALID:
            return [], np.full_like(data, np.nan)

        data = np.interp(np.arange(len(data)), np.flatnonzero(valid), data[valid])
        residue = data.copy()
        imfs: list[np.ndarray] = []

        for _ in range(max_imfs):
            if len(find_peaks(residue)[0]) + len(find_peaks(-residue)[0]) < 4:
                break
            candidate = residue.copy()
            for _ in range(max_siftings):
                maxima = find_peaks(candidate)[0]
                minima = find_peaks(-candidate)[0]
                if len(maxima) < 2 or len(minima) < 2:
                    break
                maxima = np.unique(np.r_[0, maxima, len(candidate) - 1])
                minima = np.unique(np.r_[0, minima, len(candidate) - 1])
                grid = np.arange(len(candidate))
                upper = CubicSpline(
                    maxima, candidate[maxima], bc_type="natural"
                )(grid)
                lower = CubicSpline(
                    minima, candidate[minima], bc_type="natural"
                )(grid)
                updated = candidate - (upper + lower) / 2
                sd = np.sum((candidate - updated) ** 2) / (
                    np.sum(candidate**2) + 1e-12
                )
                candidate = updated
                if sd < stop_sd:
                    break
            imfs.append(candidate)
            residue = residue - candidate

        return imfs, residue

    def _forecast_kp(self, frame: pd.DataFrame) -> pd.DataFrame:
        cuts = tuple(int(cut) for cut in self.model_bundle["ordinal_cuts"])
        models_by_bucket = self.model_bundle["models"]
        features_by_bucket = self.model_bundle["feature_columns_by_bucket"]
        calibrators_by_bucket = self.model_bundle.get("calibrators", {})

        for bucket in self.model_bundle.get(
            "horizon_buckets", DEFAULT_HORIZON_BUCKETS
        ):
            mask = frame["horizon_bucket"].eq(bucket)
            if not mask.any():
                continue

            bucket_models = self._mapping_value(models_by_bucket, bucket)
            feature_columns = list(self._mapping_value(features_by_bucket, bucket))
            for feature in feature_columns:
                if feature not in frame:
                    # Optional upstream-forecast features are deliberately safe to
                    # omit; LightGBM and the notebook's imputer both accept NaNs.
                    frame[feature] = np.nan

            raw = np.column_stack(
                [
                    self._positive_probability(
                        self._mapping_value(bucket_models, cut),
                        frame.loc[mask, feature_columns],
                    )
                    for cut in cuts
                ]
            )
            raw = np.minimum.accumulate(np.clip(raw, 0, 1), axis=1)

            bucket_calibrators = self._optional_mapping_value(
                calibrators_by_bucket, bucket
            )
            calibrated = np.column_stack(
                [
                    self._apply_probability_calibrator(
                        self._optional_mapping_value(bucket_calibrators, cut),
                        raw[:, index],
                    )
                    for index, cut in enumerate(cuts)
                ]
            )
            calibrated = np.minimum.accumulate(
                np.clip(calibrated, 0, 1), axis=1
            )

            frame.loc[mask, "kp_expected"] = calibrated.sum(axis=1).clip(0, 9)
            for index, cut in enumerate(cuts):
                frame.loc[mask, f"p_kp_{cut}"] = calibrated[:, index]

        for cut in (4, 5, 6, 7):
            column = f"p_kp_{cut}"
            if column not in frame:
                frame[column] = 0.0
            else:
                frame[column] = frame[column].fillna(0.0)
        return frame

    @staticmethod
    def _positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(features))
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive = np.flatnonzero(classes == 1)
        if not len(positive):
            return np.zeros(len(features), dtype=float)
        return probabilities[:, int(positive[0])]

    @classmethod
    def _apply_probability_calibrator(
        cls, calibrator: dict | None, probability: np.ndarray
    ) -> np.ndarray:
        raw_values = np.asarray(probability, dtype=float)
        if not calibrator:
            return np.clip(raw_values, 0, 1)
        values = np.clip(raw_values, 1e-6, 1 - 1e-6)
        if calibrator["method"] == "constant":
            return np.full(len(values), float(calibrator["constant"]))
        if calibrator["method"] == "isotonic":
            return np.asarray(calibrator["estimator"].predict(values))
        if calibrator["method"] == "platt":
            logits = np.log(values / (1 - values)).reshape(-1, 1)
            return cls._positive_probability(calibrator["estimator"], logits)
        raise ValueError(f"Unsupported probability calibrator: {calibrator['method']}")

    @staticmethod
    def _mapping_value(mapping: dict, key: Any):
        value = KpBoostedForecastService._optional_mapping_value(mapping, key)
        if value is None:
            raise KeyError(key)
        return value

    @staticmethod
    def _optional_mapping_value(mapping: dict | None, key: Any):
        if not mapping:
            return None
        if key in mapping:
            return mapping[key]
        string_key = str(key)
        if string_key in mapping:
            return mapping[string_key]
        return None
