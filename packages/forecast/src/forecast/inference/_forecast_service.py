from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from common.adapters import observations_to_dataframe
from common.schemas.forecast import Forecast, ForecastPoint
from common.schemas.observation import Observation
from forecast.quantiles import (
    apply_quantile_calibration,
    predict_overlapping_quantiles,
    uses_overlapping_buckets,
)


class DefaultForecastService(ABC):
    registry_name: str|None = None
    target_name: str|None = None
    models_bundle: dict|None = None

    def __init__(self, models_bundle: dict):
        self.models_bundle = models_bundle

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
        """Extract exatt forecasted fields from forecast dataframe row"""

    def forecast(self, observations: Observation):
        issue_time = datetime.now(UTC)

        frame = self._prepare_frame(
            observations=observations,
            issue_time=issue_time,
            lead_hours=self.models_bundle["lead_hours"]
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

    def _prepare_frame(self, observations: Observation, issue_time: datetime, lead_hours: int) -> pd.DataFrame:
        forecast_start_time = issue_time - timedelta(minutes=issue_time.minute, seconds=issue_time.second)

        df = observations_to_dataframe(observations)

        df = self._build_features(df)

        last_row = df.iloc[[-1]].copy()

        frame = pd.concat([last_row] * lead_hours, ignore_index=True)
        frame["lead_hours"] = range(1, lead_hours + 1)
        frame["valid_time"] = forecast_start_time + pd.to_timedelta(
            frame["lead_hours"], unit="h"
        )

        return frame

    def _apply_lead_buckets(self, df, lead_buckets):
        if uses_overlapping_buckets(lead_buckets):
            return

        bins = [0] + [upper for upper, _ in lead_buckets]
        labels = [label for _, label in lead_buckets]

        df["lead_bucket"] = pd.cut(
            df["lead_hours"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

class ThresholdForecastService(DefaultForecastService):
    thresholds: list

    def _build_forecast(self, frame: pd.DataFrame, models: dict, features: list) -> pd.DataFrame:
        forecast_thresholds = []

        for (threshold, lead_bucket), model in models.items():
            forecast_thresholds.append(threshold)

            column = self._col_name(threshold)

            mask = frame["lead_bucket"] == lead_bucket
            if not mask.any():
                continue

            proba = model.predict_proba(frame.loc[mask, features])

            if 1 in model.classes_:
                class_1_idx = np.where(model.classes_ == 1)[0][0]
                frame.loc[mask, column] = proba[:, class_1_idx]
            else:
                # model was trained with only class 0, so P(class 1) = 0
                frame.loc[mask, column] = 0.0

        self.thresholds = sorted(set(forecast_thresholds))
        
        return frame

    def _forecast_row(self, row) -> dict:
        forecast_row = {}

        for threshold in self.thresholds:
            column = self._col_name(threshold)
            forecast_row[column] = float(row[column])

        return forecast_row

    def _col_name(self, threshold) -> str:
        return f"p_{self.target_name}_ge_{threshold}"

class QuantileForecastService(DefaultForecastService):
    def _build_forecast(
        self,
        frame: pd.DataFrame,
        models: dict,
        features: list
    ) -> pd.DataFrame:
        buckets = self.models_bundle["buckets"]

        if uses_overlapping_buckets(buckets):
            predictions = predict_overlapping_quantiles(
                frame=frame,
                models=models,
                features=features,
                buckets=buckets,
            )
            calibration = self.models_bundle.get("calibration")
            if calibration is not None:
                predictions = apply_quantile_calibration(
                    predictions=predictions,
                    lead_hours=frame["lead_hours"].to_numpy(),
                    calibration=calibration,
                )

            for model_name in ["q10", "q50", "q90"]:
                frame.loc[:, f"{self.target_name}_{model_name}"] = predictions[
                    model_name
                ].to_numpy()
        else:
            for (lead_bucket, model_name), model in models.items():
                mask = frame["lead_bucket"].eq(lead_bucket)

                if mask.any():
                    output_column = f"{self.target_name}_{model_name}"
                    frame.loc[mask, output_column] = model.predict(
                        frame.loc[mask, features]
                    )

        quantile_columns = [
            self._col_name(10),
            self._col_name(50),
            self._col_name(90),
        ]

        frame.loc[:, quantile_columns] = np.sort(
            frame[quantile_columns].to_numpy(),
            axis=1,
        )

        return frame

    def _forecast_row(self, row) -> dict:
        forecast_row = {}

        for q in [10, 50, 90]:
            column = self._col_name(q)
            forecast_row[column] = float(row[column])

        return forecast_row

    def _col_name(self, quantile: int) -> str:
        return f"{self.target_name}_q{quantile}"
