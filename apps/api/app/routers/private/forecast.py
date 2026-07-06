from fastapi import APIRouter

from app.schemas.response import SuccessForecastResponse, ErrorResponse, SuccessResponse
from forecast.ForecastDirector import ForecastDirector
from forecast_core.inference.HmfForecastService import HmfForecastService

from common.config import get_config
import pandas as pd

router = APIRouter(prefix="/private/forecast", tags=["forecast-private"])

@router.get("/plasma")
def plasma_forecast():
    return ErrorResponse(error="Forecast not implemented.")

@router.get("/hmf")
def hmf_forecast():
    director = ForecastDirector()
    f = director.get_forecast(HmfForecastService)

    if not f:
        return ErrorResponse(error="Forecast not ready yet. Please wait.")
    
    return SuccessForecastResponse(data=f)

@router.get("/hmf/metrics")
def hmf_forecast_metrics():
    config = get_config()

    metrics = {
        "bt_threshold_5": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "bt/threshold_5.csv",
        "bt_threshold_10": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "bt/threshold_10.csv",
        "bt_threshold_15": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "bt/threshold_15.csv",
        "southward_bz_threshold_5": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "southward_bz/threshold_5.csv",
        "southward_bz_threshold_10": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "southward_bz/threshold_10.csv",
        "southward_bz_threshold_15": config.workdir / config.models_registry["models"]["hmf"]["metrics"] / "southward_bz/threshold_15.csv",
    }

    data = {}

    for title, p in metrics.items():
        df = pd.read_csv(p)
        data[title] = df.to_dict('records')

    return SuccessResponse(data=data)
