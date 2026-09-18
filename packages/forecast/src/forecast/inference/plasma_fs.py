import pandas as pd

from forecast.inference._forecast_service import (
    QuantileForecastService,
    ThresholdForecastService,
)


class SWSpeedFS(QuantileForecastService):
    registry_name: str|None = "plasma_speed_quantile"
    target_name: str|None = "v"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.features import build_features

        df = build_features(raw_observations_frame)
        return df

class SWSpeedProbaFS(ThresholdForecastService):
    registry_name: str|None = "plasma_speed_threshold"
    target_name: str|None = "v"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.features import build_features

        df = build_features(raw_observations_frame)
        return df

class SWDensityFS(QuantileForecastService):
    registry_name: str|None = "plasma_density_quantile"
    target_name: str|None = "n"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.features import build_features

        df = build_features(raw_observations_frame)
        return df
