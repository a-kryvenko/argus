from fastapi import APIRouter

from app.schemas.response import success_response, error_response
from forecast.ForecastDirector import ForecastDirector
from forecast_core.inference.HmfForecastService import HmfForecastService

from common.config import get_config
import pandas as pd

router = APIRouter(prefix="/private/forecast", tags=["forecast-private"])

@router.get("/plasma")
def plasma_forecast():
    return error_response(
        code="FORECAST_NOT_READY",
        msg="Forecast not ready yet. Please wait..."
    )

@router.get("/hmf")
def hmf_forecast():
    director = ForecastDirector()
    forecast = director.get_forecast(HmfForecastService)

    if not forecast:
        return error_response(
            code="FORECAST_NOT_READY",
            msg="Forecast not ready yet. Please wait..."
        )
    
    return success_response(forecast)

@router.get("/hmf/metrics")
def hmf_forecast_metrics():
    config = get_config()

    metrics = {
        "bt_5": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "bt/threshold_5.csv",
        "bt_10": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "bt/threshold_10.csv",
        "bt_15": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "bt/threshold_15.csv",
        "southward_bz_5": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "southward_bz/threshold_5.csv",
        "southward_bz_10": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "southward_bz/threshold_10.csv",
        "southward_bz_15": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "southward_bz/threshold_15.csv",
    }

    data = {}

    for title, p in metrics.items():
        df = pd.read_csv(p)
        data[title] = df.to_dict('records')

    return success_response(data)
