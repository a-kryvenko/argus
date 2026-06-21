import os
import csv
import re
import shutil
import joblib
import pandas as pd
from datetime import datetime, UTC

from common.config import get_config
from common.adapters import forecast_to_dataframe
from forecast.data_pipelines.live import get_live_observations

DISPLAYED_FORECAST_HORIZON = 96

class ForecastDirector:
    def get_forecast(self, forecast_service_name):
        config = get_config()

        models_registry = (config.models_registry["models"][forecast_service_name.registry_name] or None)

        if models_registry is None:
            raise Exception(f"Not found registry for {forecast_service_name.registry_name}")
        
        forecast_file_path = config.workdir / models_registry["forecast_path"]
        
        if not forecast_file_path.is_file():
            self.refresh_forecast(forecast_service_name)

        df = pd.read_csv(forecast_file_path, parse_dates=["issue_time", "valid_time"])
        now = datetime.now(UTC)

        df = (
            df[df["valid_time"] >= now]
            .head(DISPLAYED_FORECAST_HORIZON)
        )
        
        return forecast_service_name.forecast_from_df(df)
    
    def refresh_forecast(self, forecast_service_name):
        config = get_config()

        models_registry = (config.models_registry["models"][forecast_service_name.registry_name] or None)

        if models_registry is None:
            raise Exception(f"Not found registry for {forecast_service_name.registry_name}")
        
        forecast_file_path = config.workdir / models_registry["forecast_path"]

        models = dict()
        if models_registry["active_versions"] is not None:
            for k, name in models_registry["active_versions"].items():
                p = config.workdir / ("data/models/" + name + ".joblib")

                if not p.is_file():
                    raise Exception(f"{p} not exists")
                    
                models[k] = joblib.load(p)
            
        if len(models) == 0:
            raise Exception(f"Not found forecast models for {forecast_service_name.registry_name}")
        
        forecast_service = forecast_service_name(models)
            
        self._build_forecast(forecast_file_path, forecast_service)

    def _build_forecast(self, forecast_file_path, forecast_service):
        forecast_dir = forecast_file_path.parent
        archive_dir = forecast_dir / "archive"
        tmp_forecast_file_path = forecast_dir / (forecast_file_path.name + ".tmp")

        os.makedirs(forecast_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)

        if forecast_file_path.is_file():
            archive_file_name = "_"
            with open(forecast_file_path) as f:
                reader = csv.reader(f)
                row = next(reader)
                row = next(reader)
                archive_file_name = re.sub('[^0-9]', '_', row[0])
            
            archive_file_name += ".csv"

            shutil.copyfile(forecast_file_path, archive_dir / archive_file_name)
        
        forecast = forecast_service.forecast(get_live_observations())
        
        df = forecast_to_dataframe(forecast)

        df.to_csv(tmp_forecast_file_path, index=False)
        shutil.move(tmp_forecast_file_path, forecast_file_path)
    