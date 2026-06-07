from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List

class ObservationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    issue_time: datetime
    bx: float
    by: float
    bz: float
    v: float
    n: float
    t: float
    kp: float

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: List[ObservationPoint]

class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_time: datetime
    lead_hours: int
    mean_v: float
    p_10_v: float
    p_50_v: float
    p_90_v: float
    prob_v_gt_450: float
    prob_v_gt_500: float
    prob_v_gt_600: float
    prob_v_gt_700: float
    kp_risk: float

class WindSpeedForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_time: datetime
    lead_hours: int
    mean_v: float
    p_10_v: float
    p_50_v: float
    p_90_v: float

class WindThresholdForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_time: datetime
    lead_hours: int
    prob_v_gt_450: float
    prob_v_gt_500: float
    prob_v_gt_600: float
    prob_v_gt_700: float

class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_time: datetime
    points: List[ForecastPoint|WindSpeedForecastPoint|WindThresholdForecastPoint]


