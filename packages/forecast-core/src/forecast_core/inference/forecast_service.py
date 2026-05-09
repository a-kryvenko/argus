from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import joblib

from common.config import get_config
from common.schema import Observation, Forecast, ForecastPoint
from common.adapters import observations_to_dataframe

from forecast_core.data_pipelines.feature_building import build_features, BASE_FEATURE_COLUMNS

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

        points = []

        for _, row in frame.iterrows():
            point = {
                "lead_hours": int(row["lead_hours"]),
                "valid_time": pd.Timestamp(row["valid_time"]).isoformat(),
                "speed_q10": float(row["speed_q10"]),
                "speed_q50": float(row["speed_q50"]),
                "speed_q90": float(row["speed_q90"]),
                "skill_regime": self._skill_regime(int(row["lead_hours"])),
            }

            for threshold in EVENT_THRESHOLDS:
                col = f"prob_v_ge_{threshold}"
                if col in frame.columns:
                    point[col] = float(row[col])

            points.append(ForecastPoint(
                lead_hours=point["lead_hours"],
                valid_time=point["valid_time"],
                mean_v=point["speed_q50"],
                p_10_v=point["speed_q10"],
                p_50_v=point["speed_q50"],
                p_90_v=point["speed_q90"],
                prob_v_gt_450=point["prob_v_ge_450"],
                prob_v_gt_500=point["prob_v_ge_500"],
                prob_v_gt_600=point["prob_v_ge_600"],
                prob_v_gt_700=point["prob_v_ge_700"],
                kp_risk=0
            ))

        return Forecast(
            issue_time=issue_time,
            points=points
        )

    def _prepare_frame(self, observatrions: Observation, issue_time: datetime) -> pd.DataFrame:
        forecast_start_time = issue_time - timedelta(minutes=issue_time.minute, seconds=issue_time.second)

        df = observations_to_dataframe(observatrions)
        df = build_features(df)

        last_row = df.iloc[[-1]].copy()

        frame = pd.concat([last_row] * 96, ignore_index=True)
        frame["lead_hours"] = range(1, 97),
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
        frame["pred_q10"] = ordered[0]
        frame["pred_q50"] = ordered[1]
        frame["pred_q90"] = ordered[2]

        calibration = pd.read_csv(self.quantile_calibreations_path)
        frame = self._apply_interval_calibration(frame, calibration)

        return frame
    
    def _forecast_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        event_bundle = joblib.load(self.events_models_path)
        event_models = event_bundle["models"]

        for threshold, model in event_models.items():
            frame[f"prob_v_ge_{threshold}"] = model.predict_proba(
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
