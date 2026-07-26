import pandas as pd
from common.config import get_config
from forecast_core.data_pipelines.gong_pipeline import (
    extract_features as extract_gong_features,
)

from forecast.data_pipelines.feature_building import build_features
from forecast.inference._forecast_service import ThresholdForecastService


class HMFTotalProbaFS(ThresholdForecastService):
    registry_name: str|None = "hmf_total_threshold"
    target_name: str|None = "bt"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        config = get_config()
        
        df = build_features(raw_observations_frame)
        features = extract_gong_features(config.workdir / config.project_config["paths"]["live_gong"])
        df = df.merge(pd.DataFrame([features]), how="cross")
        return df

class HMFSouthProbaFS(ThresholdForecastService):
    registry_name: str|None = "hmf_southward_threshold"
    target_name: str|None = "bs"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        config = get_config()
        
        df = build_features(raw_observations_frame)
        features = extract_gong_features(config.workdir / config.project_config["paths"]["live_gong"])
        df = df.merge(pd.DataFrame([features]), how="cross")
        return df
    