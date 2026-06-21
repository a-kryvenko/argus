from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse
from forecast.ForecastDirector import ForecastDirector
from forecast.inference.PlasmaStateForecastService import PlasmaStateForecastService
from forecast.inference.KpForecastService import KpForecastService

router = APIRouter(prefix="/public/forecast", tags=["forecast-public"])

@router.get("/solar-wind")
def get_wind_forecast():
    director = ForecastDirector()
    f = director.get_forecast(PlasmaStateForecastService)

    if not f:
        return ErrorResponse(error="Forecast not ready yet. Please wait.")

    return SuccessResponse(data=f)

@router.get("/kp")
def get_kp_forecast():
    director = ForecastDirector()
    f = director.get_forecast(KpForecastService)

    if not f:
        return ErrorResponse(error="Forecast not ready yet. Please wait.")

    return SuccessResponse(data=f)
