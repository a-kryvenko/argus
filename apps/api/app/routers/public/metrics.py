from fastapi import APIRouter

from app.stats.metrics import wind_speed_metrics, wind_threshold_metrics

router = APIRouter(prefix="/public/metrics", tags=["metrics"])

@router.get("/all")
def get_full_forecast_metrics():
    f_s = wind_speed_metrics()
    f_t = wind_threshold_metrics()

    if not f_s or not f_t:
        return {"error": "Metrics not ready. Please try again later."}

    return {
        "status": "ok",
        "wind_speed": f_s,
        "wind_threshold": f_t
    }

@router.get("/wind-speed")
def get_wind_speed_metrics():
    f = wind_speed_metrics()

    if not f:
        return {"error": "Metrics not ready. Please try again later."}

    return {
        "status": "ok",
        "response": f
    }

@router.get("/wind-threshold")
def get_wind_threshold_metrics():
    f = wind_threshold_metrics()

    if not f:
        return {"error": "Metrics not ready. Please try again later."}

    return {
        "status": "ok",
        "response": f
    }