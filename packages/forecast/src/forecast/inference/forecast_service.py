from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import joblib

from common.config import get_config
from common.schema import Observation, Forecast
from common.adapters import observations_to_dataframe, wind_forecast_from_dataframe

from forecast.data_pipelines.feature_building import build_features, BASE_FEATURE_COLUMNS

EVENT_THRESHOLDS = [450, 500, 600, 700]

FORECAST_FEATURE_COLUMNS = [
    "lead_hours",
    "lead_hours_norm"
] + BASE_FEATURE_COLUMNS

class ForecastInferenceService:
    def __init__(self):
        config = get_config()

        self.quantile_models_path = config.workdir / config.project_config["models"]["wind_quantile"]
        self.quantile_calibreations_path = config.workdir / config.project_config["models"]["wind_quantile_calibration"]

        self.events_models_path = config.workdir / config.project_config["models"]["events"]

    def predict(self, observations: Observation) -> Forecast:
        issue_time = datetime.now(tz=timezone.utc) 

        frame = self._prepare_frame(observations=observations, issue_time=issue_time)        

        frame = self._forecast_wind_speed(frame)
        frame = self._forecast_events(frame)

        frame["kp_risk"] = 0

        return wind_forecast_from_dataframe(frame)

    def _prepare_frame(self, observations: Observation, issue_time: datetime) -> pd.DataFrame:
        forecast_start_time = issue_time - timedelta(minutes=issue_time.minute, seconds=issue_time.second)

        df = observations_to_dataframe(observations)
        df = build_features(df)

        last_row = df.iloc[[-1]].copy()

        frame = pd.concat([last_row] * 96, ignore_index=True)
        frame["lead_hours"] = range(1, 97)
        frame["valid_time"] = forecast_start_time + pd.to_timedelta(
            frame["lead_hours"], unit="h"
        )
        frame["lead_hours_norm"] = frame["lead_hours"] / 96
        
        return frame
    
    def _forecast_wind_speed(self, frame: pd.DataFrame) -> pd.DataFrame:
        quantile_bundle = joblib.load(self.quantile_models_path)
        quantile_models = quantile_bundle["models"]
        
        for q_name, model in quantile_models.items():
            frame[f"pred_{q_name}"] = model.predict(frame[FORECAST_FEATURE_COLUMNS])
        
        ordered = np.sort(
            np.vstack([
                frame["pred_q10"].to_numpy(),
                frame["pred_q50"].to_numpy(),
                frame["pred_q90"].to_numpy(),
            ]),
            axis=0,
        )
        frame["p_10_v"] = ordered[0]
        frame["p_50_v"] = ordered[1]
        frame["p_90_v"] = ordered[2]

        calibration = pd.read_csv(self.quantile_calibreations_path)
        frame = self._apply_interval_calibration(frame, calibration)

        return frame
    
    def _forecast_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        event_bundle = joblib.load(self.events_models_path)
        event_models = event_bundle["models"]

        for threshold, model in event_models.items():
            frame[f"prob_v_gt_{threshold}"] = model.predict_proba(
                frame[FORECAST_FEATURE_COLUMNS]
            )[:, 1]
        
        return frame
    
    def _apply_interval_calibration(self, df: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(
            calibration[["lead_hours", "scale"]],
            on="lead_hours",
            how="left",
        )

        out["scale"] = out["scale"].fillna(1.0)

        out["speed_q50"] = out["pred_q50"]

        out["speed_q10"] = (
            out["pred_q50"]
            - out["scale"] * (out["pred_q50"] - out["pred_q10"])
        )

        out["speed_q90"] = (
            out["pred_q50"]
            + out["scale"] * (out["pred_q90"] - out["pred_q50"])
        )

        return out

    def _skill_regime(self, lead_hours: int) -> str:
        if lead_hours <= 24:
            return "strong"
        if lead_hours <= 48:
            return "moderate"
        return "low"
