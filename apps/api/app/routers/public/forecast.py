from fastapi import APIRouter

from forecast.predictor import wind_forecast
from app.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/public/forecast", tags=["forecast-public"])

@router.get("/solar-wind")
def solar_wind_forecast():
    f = wind_forecast()

    if not f:
        return ErrorResponse(error="Forecast not ready yet. Please wait.")

    return SuccessResponse(data=f)

@router.get("/kp")
def kp_forecast():
    return ErrorResponse(error="Forecast not implemented.")
