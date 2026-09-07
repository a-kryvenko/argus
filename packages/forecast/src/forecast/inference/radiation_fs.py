import pandas as pd

from forecast.inference._forecast_service import QuantileForecastService


class F107FS(QuantileForecastService):
    registry_name: str|None = "f10_7_quantile"
    target_name: str|None = "f10_7"
    

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast_core.api import build_features

        df = build_features(raw_observations_frame)
        return df

class S10FS(QuantileForecastService):
    registry_name: str|None = "s10_quantile"
    target_name: str|None = "s10"
    

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast_core.api import build_features

        df = build_features(raw_observations_frame)
        return df

class M10FS(QuantileForecastService):
    registry_name: str|None = "m10_quantile"
    target_name: str|None = "m10"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast_core.api import build_features

        df = build_features(raw_observations_frame)
        return df

class Y10FS(QuantileForecastService):
    registry_name: str|None = "y10_quantile"
    target_name: str|None = "y10"
    

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast_core.api import build_features

        df = build_features(raw_observations_frame)
        return df
