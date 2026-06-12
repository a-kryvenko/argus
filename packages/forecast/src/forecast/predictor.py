import pandas as pd
from pathlib import Path
import shutil
import csv
import re
import os

from forecast.inference.forecast_service import ForecastInferenceService
from common.config import get_config
from common.adapters import wind_forecast_to_dataframe, wind_forecast_from_dataframe
from common.schema import Forecast
from forecast.data_pipelines.live import get_live_observations

def wind_forecast() -> Forecast:
    return wind_forecast_from_dataframe(_get_forecast_df())

def _get_forecast_df() -> pd.DataFrame:
    config = get_config()

    forecast_path = config.workdir / config.project_config["paths"]["wind_forecast"]

    if not forecast_path.is_file():
        _create_forecast(forecast_path)
        
    return pd.read_csv(forecast_path, parse_dates=["issue_time", "valid_time"])

def refresh_forecast():
    config = get_config()

    forecast_path = config.workdir / config.project_config["paths"]["wind_forecast"]
    forecast_path_tmp = config.workdir / (config.project_config["paths"]["wind_forecast"] + ".tmp")
    forecast_history = config.workdir / config.project_config["paths"]["forecast_history"]

    os.makedirs(forecast_history, exist_ok=True)

    if forecast_path.is_file():
        new_name = "_"
        with open(forecast_path) as f:
            reader = csv.reader(f)
            row = next(reader)
            row = next(reader)
            new_name = re.sub('[^0-9]', '_', row[0])
        
        new_name += ".csv"

        shutil.copyfile(forecast_path, forecast_history / new_name)
    
    _create_forecast(forecast_path_tmp)
    shutil.move(forecast_path_tmp, forecast_path)


def _create_forecast(output_path: Path):
    os.makedirs(output_path.parent, exist_ok=True)

    forecast_service = ForecastInferenceService()

    forecast = forecast_service.predict(get_live_observations())

    df = wind_forecast_to_dataframe(forecast)
    df.insert(0, "issue_time", forecast.issue_time)
    df.to_csv(output_path, index=False)
