from common.schemas.forecast import ForecastPoint
from typing import Dict

class SateliteDragRiskForecastPoint(ForecastPoint):
    drag_risk: float
    p_elevated_drag: float

class SateliteChargingRiskForecastPoint(ForecastPoint):
    charging_risk: float
    drivers: Dict