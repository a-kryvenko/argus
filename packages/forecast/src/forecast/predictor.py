import pandas as pd
from pathlib import Path
import shutil
import csv
import re
import os

from forecast.inference.forecast_service import ForecastInferenceService
from common.config import get_config
from common.adapters import forecast_to_dataframe, plasma_forecast_from_dataframe, kp_forecast_from_dataframe
from common.schema import Forecast
from forecast.data_pipelines.live import get_live_observations

def wind_forecast() -> Forecast:
    config = get_config()

    return plasma_forecast_from_dataframe(_get_forecast_df(config.workdir / config.project_config["paths"]["wind_forecast"]))

def kp_forecast() -> Forecast:
    config = get_config()

    return kp_forecast_from_dataframe(_get_forecast_df(config.workdir / config.project_config["paths"]["kp_forecast"]))

def _get_forecast_df(path: Path) -> pd.DataFrame:
    if not path.is_file():
        refresh_forecast()
        
    return pd.read_csv(path, parse_dates=["issue_time", "valid_time"])

def refresh_forecast():
    config = get_config()

    forecasts_config = {
        "plasma": config.workdir / config.project_config["paths"]["wind_forecast"],
        "kp": config.workdir / config.project_config["paths"]["kp_forecast"],
    }

    for k, p in forecasts_config.items():
        if p.is_file():
            new_name = "_"
            with open(p) as f:
                reader = csv.reader(f)
                row = next(reader)
                row = next(reader)
                new_name = re.sub('[^0-9]', '_', row[0])
            
            new_name += ".csv"

            history_dir = p.parent / "archive"
            os.makedirs(history_dir, exist_ok=True)
            shutil.copyfile(p, history_dir / new_name)
    
    forecasts = _request_forecast()
    for k, p in forecasts_config.items():
        os.makedirs(p.parent, exist_ok=True)

        tmp_path = p.parent / (p.name + ".tmp")

        df = forecast_to_dataframe(forecasts[k])
        df.insert(0, "issue_time", forecasts[k].issue_time)

        df.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, p)

def _request_forecast():
    forecast_service = ForecastInferenceService()
    return forecast_service.predict(get_live_observations())
