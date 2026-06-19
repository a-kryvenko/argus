from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import List, Dict

class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_time: datetime
    lead_hours: int

class WindForecastPoint(ForecastPoint):
    v_q10: float
    v_q50: float
    v_q90: float
    p_v_ge_450: float
    p_v_ge_500: float
    p_v_ge_600: float

class KpForecastPoint(ForecastPoint):
    p_kp_4: float
    p_kp_5: float
    p_kp_6: float
    p_kp_7: float

class BzForecastPoint(ForecastPoint):
    p_bz_lt_0: float
    p_bz_lt_minus_5: float
    p_bz_lt_minus_10: float
    p_bz_lt_minus_15: float
    confidence: float

class ImfForecastPoint(ForecastPoint):
    bt_q10: float
    bt_q50: float
    bt_q90: float
    p_bt_gt_10: float
    p_bt_gt_15: float

class PlasmaForecastPoint(ForecastPoint):
    n_q10: float
    n_q50: float
    n_q90: float
    dynamic_pressure_q50: float
    p_dynamic_pressure_gt_5: float

class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_time: datetime
    query: Dict = Field(default_factory=dict)
    points: List[WindForecastPoint|KpForecastPoint|BzForecastPoint|ImfForecastPoint|PlasmaForecastPoint]


