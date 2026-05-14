from fastapi import APIRouter

from forecast_core.predictor import get_forecast

router = APIRouter(prefix="/api", tags=["api"])

@router.get("/forecast")
def get_solar_wind():
    f = get_forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        "status": "ok",
        "last_update": f.issue_time,
        "forecast": f.points
    }

@router.get("/health")
def health():
    f = get_forecast()

    return {
        "status": "ok",
        "last_update": f.issue_time if f else None
    }
