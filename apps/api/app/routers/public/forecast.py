from fastapi import APIRouter

from app.schemas.response import SuccessForecastResponse, ErrorResponse, SuccessResponse
from forecast.ForecastDirector import ForecastDirector
from forecast.inference.PlasmaStateForecastService import PlasmaStateForecastService
from forecast.inference.KpForecastService import KpForecastService

from common.config import get_config
import pandas as pd

router = APIRouter(prefix="/public/forecast", tags=["forecast-public"])

@router.get("/solar-wind")
def get_wind_forecast():
    director = ForecastDirector()
    f = director.get_forecast(PlasmaStateForecastService)

    if not f:
        return ErrorResponse(error="Forecast not ready yet. Please wait.")

    return SuccessForecastResponse(data=f)

@router.get("/solar-wind/metrics")
def wind_forecast_metrics():
    config = get_config()

    metrics = {
        "v_450": config.workdir / config.models_registry["models"]["solar_wind_speed"]["metrics"] / "threshold_450.csv",
        "v_500": config.workdir / config.models_registry["models"]["solar_wind_speed"]["metrics"] / "threshold_500.csv",
        "v_600": config.workdir / config.models_registry["models"]["solar_wind_speed"]["metrics"] / "threshold_600.csv",
        "regression": config.workdir / config.models_registry["models"]["solar_wind_speed"]["metrics"] / "regression.csv",
    }

    data = {}

    for title, p in metrics.items():
        df = pd.read_csv(p)
        data[title] = df.to_dict('records')

    return SuccessResponse(data=data)

@router.get("/kp")
def get_kp_forecast():
    director = ForecastDirector()
    f = director.get_forecast(KpForecastService)

    if not f:
        return ErrorResponse(error="Forecast not ready yet. Please wait.")

    return SuccessForecastResponse(data=f)

@router.get("/kp/metrics")
def kp_forecast_metrics():
    config = get_config()

    metrics = {
        "kp_4": config.workdir / config.models_registry["models"]["kp"]["metrics"] / "threshold_4.csv",
        "kp_5": config.workdir / config.models_registry["models"]["kp"]["metrics"] / "threshold_5.csv",
        "kp_6": config.workdir / config.models_registry["models"]["kp"]["metrics"] / "threshold_6.csv",
    }

    data = {}

    for title, p in metrics.items():
        df = pd.read_csv(p)
        data[title] = df.to_dict('records')

    return SuccessResponse(data=data)

@router.get("/ap/metrics")
def ap_forecast_metrics():
    config = get_config()

    metrics = {
        "ap": config.workdir / config.models_registry["models"]["ap"]["metrics"] / "regression.csv",
    }

    data = {}

    for title, p in metrics.items():
        df = pd.read_csv(p)
        data[title] = df.to_dict('records')

    return SuccessResponse(data=data)