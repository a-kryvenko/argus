from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from common.adapters import kp_forecast_from_dataframe, observations_to_dataframe
from common.schema import Observation
from forecast.data_pipelines.feature_building import build_features

LEAD_HOURS_HORIZON = 120

class KpForecastService:
    registry_name: str = "kp"

    threshold_model_bundle = None

    def __init__(self, models: dict):
        self.threshold_model_bundle = models["threshold"]
    
    def forecast_from_df(df):
        return kp_forecast_from_dataframe(df)
    
    def forecast(self, observations: Observation):
        issue_time = datetime.now(tz=timezone.utc) 

        frame = self._prepare_frame(observations=observations, issue_time=issue_time)        

        frame = self._forecast_kp_threshold(frame)

        return KpForecastService.forecast_from_df(frame)

    def _prepare_frame(self, observations: Observation, issue_time: datetime) -> pd.DataFrame:
        forecast_start_time = issue_time - timedelta(minutes=issue_time.minute, seconds=issue_time.second)

        df = observations_to_dataframe(observations)
        df = build_features(df)

        last_row = df.iloc[[-1]].copy()

        frame = pd.concat([last_row] * LEAD_HOURS_HORIZON, ignore_index=True)
        frame["lead_hours"] = range(1, LEAD_HOURS_HORIZON + 1)
        frame["valid_time"] = forecast_start_time + pd.to_timedelta(
            frame["lead_hours"], unit="h"
        )
        frame["lead_norm"] = frame["lead_hours"] / LEAD_HOURS_HORIZON

        return frame
    
    def _forecast_kp_threshold(self, frame: pd.DataFrame) -> pd.DataFrame:
        event_bundle = self.threshold_model_bundle
        event_models = event_bundle["models"]
        event_features = event_bundle["feature_columns"]

        for threshold, model in event_models.items():
            col = f"p_kp_{threshold}"

            proba = model.predict_proba(
                frame[event_features]
            )

            if 1 in model.classes_:
                class_1_idx = np.where(model.classes_ == 1)[0][0]
                frame[col] = proba[:, class_1_idx]
            else:
                frame[col] = 0.0
        
        return frame