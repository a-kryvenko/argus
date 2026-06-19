from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict
import pandas as pd
import numpy as np
import joblib

from common.config import get_config
from common.schema import Observation
from common.adapters import observations_to_dataframe, plasma_forecast_from_dataframe, kp_forecast_from_dataframe

from forecast.data_pipelines.feature_building import build_features

PLASMA_SPEED_THRESHOLDS = [450, 500, 600, 700]
KP_THRESHOLD = [4, 5, 6]

class ForecastInferenceService:
    def __init__(self):
        config = get_config()

        self.plasma_speed_model = config.workdir / config.project_config["models"]["plasma_speed"]
        self.plasma_threshold_model = config.workdir / config.project_config["models"]["plasma_threshold"]
        self.kp_threshold_model = config.workdir / config.project_config["models"]["kp_threshold"]

    def predict(self, observations: Observation) -> Dict:
        issue_time = datetime.now(tz=timezone.utc) 

        frame = self._prepare_frame(observations=observations, issue_time=issue_time)        

        frame = self._forecast_plasma_speed(frame)
        frame = self._forecast_plasma_threshold(frame)
        frame = self._forecast_kp_threshold(frame)

        return {
            "plasma": plasma_forecast_from_dataframe(frame),
            "kp": kp_forecast_from_dataframe(frame)
        }

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
        frame["lead_norm"] = frame["lead_hours"] / 96

        def assign_bucket(lead_hours: int) -> str:
            if lead_hours <= 3:
                return "short_1_3"
                
            if lead_hours <= 6:
                return "short_4_6"

            if lead_hours <= 12:
                return "short_7_12"

            if lead_hours <= 24:
                return "medium_13_24"

            if lead_hours <= 36:
                return "medium_25_36"

            if lead_hours <= 48:
                return "long_37_48"

            if lead_hours <= 60:
                return "long_49_60"

            return "extended_61_96"

        frame["lead_bucket"] = frame["lead_hours"].apply(assign_bucket)
        
        return frame
    
    def _forecast_plasma_speed(self, frame: pd.DataFrame) -> pd.DataFrame:
        model_bundle = joblib.load(self.plasma_speed_model)
        calibration = model_bundle["calibration"]
        quantile_models = model_bundle["models"]
        quantile_features = model_bundle["feature_columns"]

        def add_smooth_quantile_predictions(df):
            out = df.copy()

            bucket_ranges = {
                "short_1_3": (1, 3),
                "short_4_6": (4, 6),
                "short_7_12": (7, 12),
                "medium_13_24": (13, 24),
                "medium_25_36": (25, 36),
                "long_37_48": (37, 48),
                "long_49_60": (49, 60),
                "extended_61_96": (61, 120),
            }

            sigmas = {
                "short_1_3": 1.5,
                "short_4_6": 1.5,
                "short_7_12": 3,
                "medium_13_24": 6,
                "medium_25_36": 6,
                "long_37_48": 6,
                "long_49_60": 5,
                "extended_61_96": 30,
            }

            bucket_centers = {
                bucket: (lo + hi) / 2
                for bucket, (lo, hi) in bucket_ranges.items()
            }

            for name in {name for _, name in quantile_models.keys()}:
                num = 0.0
                den = 0.0

                for (bucket, q_name), model in quantile_models.items():
                    if q_name != name:
                        continue

                    center = bucket_centers[bucket]
                    sigma = sigmas[bucket]

                    pred = model.predict(out[quantile_features])

                    weight = np.exp(
                        -0.5 * ((out["lead_hours"] - center) / sigma) ** 2
                    )

                    max_dist = 2.5 * sigma
                    weight = weight.where(
                        (out["lead_hours"] - center).abs() <= max_dist,
                        0.0,
                    )

                    num += weight * pred
                    den += weight

                out[f"pred_v_{name}"] = num / den
                    
            q = np.sort(
                np.vstack([
                    out["pred_v_q10"].to_numpy(),
                    out["pred_v_q50"].to_numpy(),
                    out["pred_v_q90"].to_numpy(),
                ]),
                axis=0,
            )

            out["pred_v_q10"] = q[0]
            out["pred_v_q50"] = q[1]
            out["pred_v_q90"] = q[2]

            return out

        def add_quantile_predictions(df):
            out = df.copy()

            for (bucket, name), model in quantile_models.items():
                mask = out["lead_bucket"] == bucket
                if not mask.any():
                    continue

                out.loc[mask, f"pred_v_{name}"] = model.predict(out.loc[mask, quantile_features])
                    
            q = np.sort(
                np.vstack([
                    out["pred_v_q10"].to_numpy(),
                    out["pred_v_q50"].to_numpy(),
                    out["pred_v_q90"].to_numpy(),
                ]),
                axis=0,
            )

            out["pred_v_q10"] = q[0]
            out["pred_v_q50"] = q[1]
            out["pred_v_q90"] = q[2]

            return out

        def apply_calibration(df):
            out = df.merge(calibration[["lead_hours", "scale"]], on="lead_hours", how="left")

            out["pred_v_q10"] = (
                out["pred_v_q50"]
                - out["scale"] * (out["pred_v_q50"] - out["pred_v_q10"])
            )

            out["pred_v_q90"] = (
                out["pred_v_q50"]
                + out["scale"] * (out["pred_v_q90"] - out["pred_v_q50"])
            )

            out = out.drop(columns=['scale'])

            return out

        #frame = add_quantile_predictions(frame)
        frame = add_smooth_quantile_predictions(frame)
        frame = apply_calibration(frame)
        
        ordered = np.sort(
            np.vstack([
                frame["pred_v_q10"].to_numpy(),
                frame["pred_v_q50"].to_numpy(),
                frame["pred_v_q90"].to_numpy(),
            ]),
            axis=0,
        )
        frame["v_q10"] = ordered[0]
        frame["v_q50"] = ordered[1]
        frame["v_q90"] = ordered[2]

        return frame
    
    def _forecast_plasma_threshold(self, frame: pd.DataFrame) -> pd.DataFrame:
        event_bundle = joblib.load(self.plasma_threshold_model)
        event_models = event_bundle["models"]
        event_features = event_bundle["feature_columns"]

        for (threshold, lead_bucket), model in event_models.items():
            mask = frame["lead_bucket"] == lead_bucket
            col_legacy = f"pred_p_v_ge_{threshold}"
            col = f"p_v_ge_{threshold}"

            if not mask.any():
                continue

            proba = model.predict_proba(
                frame.loc[mask, event_features]
            )

            if 1 in model.classes_:
                class_1_idx = np.where(model.classes_ == 1)[0][0]
                frame.loc[mask, col_legacy] = proba[:, class_1_idx]
                frame.loc[mask, col] = proba[:, class_1_idx]
            else:
                frame.loc[mask, col_legacy] = 0.0
                frame.loc[mask, col] = 0.0
        
        return frame
    
    def _forecast_kp_threshold(self, frame: pd.DataFrame) -> pd.DataFrame:
        event_bundle = joblib.load(self.kp_threshold_model)
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

    def _skill_regime(self, lead_hours: int) -> str:
        if lead_hours <= 24:
            return "strong"
        if lead_hours <= 48:
            return "moderate"
        return "low"
        
