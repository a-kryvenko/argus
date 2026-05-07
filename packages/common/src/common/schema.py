from pydantic import BaseModel
from datetime import datetime

class Forecast(BaseModel):
    mean_v: float
    std_v: float
    p10_v: float
    p50_v: float
    p90_v: float
    prob_v_gt_450: float
    prob_v_gt_500: float
    prob_v_gt_600: float
    prob_v_gt_700: float
    prob_v_gt_800: float
    kp_risk_proxy: float

class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast: Forecast
