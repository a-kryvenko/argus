"""Shared forecast lifecycle with explicitly supplied feature models."""
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd
from common.adapters import observations_to_dataframe
from common.schemas.forecast import Forecast, ForecastPoint
from common.schemas.observation import Observation


class FeatureModel(Protocol):
    """Prepare raw history before feature construction, then enrich horizons."""

    def snapshot_options(self, inputs) -> dict: ...

    def prepare(self, observations: pd.DataFrame, *, issue_time: datetime,
                speed_history: pd.DataFrame | None = None) -> pd.DataFrame: ...

    def apply(self, frame: pd.DataFrame, history: pd.DataFrame, *,
              issue_time: datetime) -> None: ...


class DefaultForecastService(ABC):
    registry_name: str|None = None
    target_name: str|None = None
    models_bundle: dict|None = None

    def __init__(self, models_bundle: dict, *, feature_models: tuple[FeatureModel, ...] = ()):
        self.models_bundle = models_bundle
        self.feature_models = tuple(feature_models)

    def snapshot_options(self, inputs):
        options = {}
        for model in self.feature_models:
            options.update(model.snapshot_options(inputs))
        return options

    def forecast_snapshot(self, inputs, *, issue_time):
        from common.adapters import forecast_to_dataframe
        return forecast_to_dataframe(self.forecast(
            inputs.observations, issue_time=issue_time, **self.snapshot_options(inputs)))

    @abstractmethod
    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        """Build required forecast features from raw observations"""

    @abstractmethod
    def _build_forecast(self, frame: pd.DataFrame, models: dict, features: list) -> pd.DataFrame:
        """Create forecast"""

    def forecast_from_df(self, df: pd.DataFrame):
        points = []
        
        for _, row in df.iterrows():
            points.append(ForecastPoint(
                lead_hours=int(row["lead_hours"]),
                valid_time=pd.Timestamp(row["valid_time"]).isoformat(),
                **self._forecast_row(row)
            ))

        return Forecast(
            issue_time=pd.Timestamp(df.iloc[0]["issue_time"]).isoformat(),
            points=points
        )

    @abstractmethod
    def _forecast_row(self, row) -> dict:
        """Extract forecast fields from a forecast dataframe row."""

    def forecast(self, observations: Observation, *, issue_time: datetime | None = None,
                 speed_history: pd.DataFrame | None = None, feature_inputs: dict | None = None):
        issue_time = issue_time or datetime.now(UTC)
        if issue_time.tzinfo is None or issue_time.utcoffset() is None:
            raise ValueError("Forecast issue_time must be timezone-aware")
        issue_time = issue_time.astimezone(UTC)

        extra = {'speed_history': speed_history} if speed_history is not None else {}
        if feature_inputs is not None:
            extra['feature_inputs'] = feature_inputs
        frame = self._prepare_frame(
            observations=observations,
            issue_time=issue_time,
            lead_hours=self.models_bundle["lead_hours"],
            **extra,
        )

        self._apply_lead_buckets(
            df=frame,
            lead_buckets=self.models_bundle["buckets"]
        )

        frame = self._build_forecast(
            frame=frame,
            models=self.models_bundle["models"],
            features=self.models_bundle["feature_columns"]
        )

        return self.forecast_from_df(frame)

    def _prepare_frame(self, observations: Observation, issue_time: datetime, lead_hours: int,
                       speed_history: pd.DataFrame | None = None, feature_inputs: dict | None = None) -> pd.DataFrame:
        forecast_start_time = issue_time.replace(minute=0, second=0, microsecond=0)

        df = observations_to_dataframe(observations)
        prepared = [model.prepare(df, issue_time=forecast_start_time, speed_history=speed_history)
                    for model in self.feature_models]

        df = self._build_features(df, **(feature_inputs or {}))

        last_row = df.iloc[[-1]].copy()

        frame = pd.concat([last_row] * lead_hours, ignore_index=True)
        frame["issue_time"] = issue_time
        frame["lead_hours"] = range(1, lead_hours + 1)
        frame["valid_time"] = forecast_start_time + pd.to_timedelta(
            frame["lead_hours"], unit="h"
        )

        for model, history in zip(self.feature_models, prepared):
            model.apply(frame, history, issue_time=forecast_start_time)

        return frame

    def _apply_lead_buckets(self, df, lead_buckets):
        """Hook for prediction mechanisms that group forecast horizons."""
