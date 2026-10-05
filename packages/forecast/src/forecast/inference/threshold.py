"""Threshold-exceedance probabilities from fitted classifiers."""
import numpy as np
import pandas as pd

from forecast.inference.bucketed import BucketedForecastService


class ThresholdForecastService(BucketedForecastService):
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
