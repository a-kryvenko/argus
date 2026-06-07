import pandas as pd
from common.schema import Observation, Forecast, ForecastPoint, WindSpeedForecastPoint, WindThresholdForecastPoint

def observations_to_dataframe(observation: Observation) -> pd.DataFrame:
    return pd.DataFrame([o.model_dump() for o in observation.points])

def forecast_to_dataframe(forecast: Forecast) -> pd.DataFrame:
    return pd.DataFrame([f.model_dump() for f in forecast.points])

def full_forecast_from_dataframe(df: pd.DataFrame) -> Forecast:
    points = []

    for _, row in df.iterrows():

        points.append(ForecastPoint(
            lead_hours=int(row["lead_hours"]),
            valid_time=pd.Timestamp(row["valid_time"]).isoformat(),
            mean_v=float(row["p_50_v"]),
            p_10_v=float(row["p_10_v"]),
            p_50_v=float(row["p_50_v"]),
            p_90_v=float(row["p_90_v"]),
            prob_v_gt_450=float(row["prob_v_gt_450"]),
            prob_v_gt_500=float(row["prob_v_gt_500"]),
            prob_v_gt_600=float(row["prob_v_gt_600"]),
            prob_v_gt_700=float(row["prob_v_gt_700"]),
            kp_risk=float(row["kp_risk"])
        ))

    return Forecast(
        issue_time=pd.Timestamp(df.iloc[0]["issue_time"]).isoformat(),
        points=points
    )

def wind_speed_forecast_from_dataframe(df: pd.DataFrame) -> Forecast:
    points = []

    for _, row in df.iterrows():

        points.append(WindSpeedForecastPoint(
            lead_hours=int(row["lead_hours"]),
            valid_time=pd.Timestamp(row["valid_time"]).isoformat(),
            mean_v=float(row["p_50_v"]),
            p_10_v=float(row["p_10_v"]),
            p_50_v=float(row["p_50_v"]),
            p_90_v=float(row["p_90_v"])
        ))

    return Forecast(
        issue_time=pd.Timestamp(df.iloc[0]["issue_time"]).isoformat(),
        points=points
    )

def wind_threshold_forecast_from_dataframe(df: pd.DataFrame) -> Forecast:
    points = []

    for _, row in df.iterrows():

        points.append(WindThresholdForecastPoint(
            lead_hours=int(row["lead_hours"]),
            valid_time=pd.Timestamp(row["valid_time"]).isoformat(),
            prob_v_gt_450=float(row["prob_v_gt_450"]),
            prob_v_gt_500=float(row["prob_v_gt_500"]),
            prob_v_gt_600=float(row["prob_v_gt_600"]),
            prob_v_gt_700=float(row["prob_v_gt_700"]),
        ))

    return Forecast(
        issue_time=pd.Timestamp(df.iloc[0]["issue_time"]).isoformat(),
        points=points
    )