import pandas as pd

from forecast.data_pipelines.feature_building import build_features
from forecast.inference._forecast_service import (
    QuantileForecastService,
    ThresholdForecastService,
)


class KPProbaFS(ThresholdForecastService):
    registry_name: str|None = "kp_threshold"
    target_name: str|None = "kp"
    
    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        df = build_features(raw_observations_frame)
        return df

class APFS(QuantileForecastService):
    registry_name: str|None = "ap_quantile"
    target_name: str|None = "ap"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        df = build_features(raw_observations_frame)
        return df

class DstFS(QuantileForecastService):
    registry_name: str|None = "dst_quantile"
    target_name: str|None = "dst"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        df = build_features(raw_observations_frame)
        return df