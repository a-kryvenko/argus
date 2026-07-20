from fastapi import APIRouter

from app.schemas.response import success_response, error_response

from app.stats.metrics import wind_speed_metrics, wind_threshold_metrics

router = APIRouter(prefix="/public/metrics", tags=["metrics"])

@router.get("/all")
def get_full_forecast_metrics():
    metrics_speed = wind_speed_metrics()
    metrics_threshold = wind_threshold_metrics()

    if not metrics_speed or not metrics_threshold:
        return error_response(
            code="METRICS_NOT_READY",
            msg="Metrics not ready. Please try again later."
        )

    return success_response({
        "wind_speed": metrics_speed,
        "wind_threshold": metrics_threshold
    })

@router.get("/wind-speed")
def get_wind_speed_metrics():
    metrics_speed = wind_speed_metrics()

    if not metrics_speed:
        return error_response(
            code="METRICS_NOT_READY",
            msg="Metrics not ready. Please try again later."
        )

    return success_response({
        "wind_speed": metrics_speed
    })

@router.get("/wind-threshold")
def get_wind_threshold_metrics():
    metrics_threshold = wind_threshold_metrics()

    if not metrics_threshold:
        return error_response(
            code="METRICS_NOT_READY",
            msg="Metrics not ready. Please try again later."
        )

    return success_response({
        "wind_threshold": metrics_threshold
    })
