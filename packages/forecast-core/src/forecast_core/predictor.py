import pandas as pd
from pathlib import Path
import shutil
import csv
import re
import os

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

def refresh_forecast():
    config = get_config()

    forecast_path = config.workdir / config.project_config["paths"]["wind_forecast"]
    forecast_path_tmp = config.workdir / (config.project_config["paths"]["wind_forecast"] + ".tmp")
    forecast_history = config.workdir / config.project_config["paths"]["forecast_history"]

    if not forecast_history.is_dir():
        os.mkdir(forecast_history)

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
    forecast_service = ForecastInferenceService()

    forecast = forecast_service.predict(get_live_observations())

    df = forecast_to_dataframe(forecast)
    df.insert(0, "issue_time", forecast.issue_time)
    df.to_csv(output_path, index=False)
