from fastapi import APIRouter

from forecast_core.predictor import get_forecast

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/all")
def get_full_forecast_metrics():
    f = get_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }

@router.get("/wind-speed")
def get_wind_speed_metrics():
    f = get_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }
@router.get("/wind-threshold")
def get_wind_threshold_metrics():
    f = get_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }