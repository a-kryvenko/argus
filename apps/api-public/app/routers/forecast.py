from fastapi import APIRouter

from forecast_core.predictor import get_full_forecast, get_wind_speed_forecast, get_wind_threshold_forecast

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/all")
def full_forecast():
    f = get_full_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }

@router.get("/wind-speed")
def wind_speed_forecast():
    f = get_wind_speed_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }
@router.get("/wind-threshold")
def wind_threshold_forecast():
    f = get_wind_threshold_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }