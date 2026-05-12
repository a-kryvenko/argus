import pandas as pd
from pathlib import Path

from forecast_core.inference.forecast_service import ForecastInferenceService
from common.config import get_config
from common.adapters import forecast_to_dataframe, forecast_from_dataframe
from common.schema import Forecast
from forecast_core.data_pipelines.live import get_live_observations

def get_forecast() -> Forecast:
    config = get_config()

    forecast_path = config.workdir / config.project_config["paths"]["wind_forecast"]

    if not forecast_path.is_file():
        _create_forecast(forecast_path)
        
    df = pd.read_csv(forecast_path, parse_dates=["issue_time", "valid_time"])
    
    return forecast_from_dataframe(df)

def _create_forecast(output_path: Path):
    forecast_service = ForecastInferenceService()

    forecast = forecast_service.predict(get_live_observations())

    df = forecast_to_dataframe(forecast)
    df.insert(0, "issue_time", forecast.issue_time)
    df.to_csv(output_path, index=False)
