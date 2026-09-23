"""In-memory forecast calculations, independent of storage and configuration."""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from common.adapters import forecast_to_dataframe
from common.schemas.observation import Observation
from forecast.inference._forecast_service import DefaultForecastService


@dataclass
class ForecastResult:
    name: str
    frame: pd.DataFrame
    model_info: dict


def calculate_forecast(service: DefaultForecastService, observations: Observation, *,
                       issue_time: datetime, model_info: dict,
                       speed_history: pd.DataFrame | None = None,
                       aia_features: pd.DataFrame | None = None) -> ForecastResult:
    extra = {'speed_history': speed_history} if speed_history is not None else {}
    if aia_features is not None:
        extra['aia_features'] = aia_features
    forecast = service.forecast(observations, issue_time=issue_time, **extra)
    return ForecastResult(service.registry_name, forecast_to_dataframe(forecast), model_info)
