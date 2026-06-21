import pandas as pd
from common.schemas.forecast import Forecast, WindForecastPoint, KpForecastPoint
from common.schema import Observation

def observations_to_dataframe(observation: Observation) -> pd.DataFrame:
    return pd.DataFrame([o.model_dump() for o in observation.points])

def forecast_to_dataframe(forecast: Forecast) -> pd.DataFrame:
    df = pd.DataFrame([f.model_dump() for f in forecast.points])
    df.insert(0, "issue_time", forecast.issue_time)
    return df


def plasma_forecast_from_dataframe(df: pd.DataFrame) -> Forecast:
    points = []

    for _, row in df.iterrows():
        points.append(WindForecastPoint(
            lead_hours=int(row["lead_hours"]),
            valid_time=pd.Timestamp(row["valid_time"]).isoformat(),
            v_q10=float(row["v_q10"]),
            v_q50=float(row["v_q50"]),
            v_q90=float(row["v_q90"]),
            p_v_ge_450=float(row["p_v_ge_450"]),
            p_v_ge_500=float(row["p_v_ge_500"]),
            p_v_ge_600=float(row["p_v_ge_600"])
        ))

    return Forecast(
        issue_time=pd.Timestamp(df.iloc[0]["issue_time"]).isoformat(),
        points=points
    )

def kp_forecast_from_dataframe(df: pd.DataFrame) -> Forecast:
    points = []

    for _, row in df.iterrows():
        points.append(KpForecastPoint(
            lead_hours=int(row["lead_hours"]),
            valid_time=pd.Timestamp(row["valid_time"]).isoformat(),
            p_kp_4=float(row["p_kp_4"]),
            p_kp_5=float(row["p_kp_5"]),
            p_kp_6=float(row["p_kp_6"]),
            p_kp_7=float(row["p_kp_7"]),
        ))

    return Forecast(
        issue_time=pd.Timestamp(df.iloc[0]["issue_time"]).isoformat(),
        points=points
    )

