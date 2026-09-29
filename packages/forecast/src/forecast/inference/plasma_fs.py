from forecast.inference.aia_wind import AIAWindServiceMixin
from forecast.inference.density_dlinear import DensityDLinearForecaster
from datetime import UTC, datetime
import pandas as pd

from forecast.inference._forecast_service import (
    QuantileForecastService,
    ThresholdForecastService,
)


class SWSpeedFS(AIAWindServiceMixin, QuantileForecastService):
    registry_name: str|None = "plasma_speed_quantile"
    target_name: str|None = "v"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.features import build_features

        df = build_features(raw_observations_frame)
        return df

class SWSpeedProbaFS(AIAWindServiceMixin, ThresholdForecastService):
    registry_name: str|None = "plasma_speed_threshold"
    target_name: str|None = "v"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.features import build_features

        df = build_features(raw_observations_frame)
        return df

class SWDensityFS(QuantileForecastService):
    registry_name: str|None = "plasma_density_quantile"
    target_name: str|None = "n"

    def __init__(self, models_bundle):
        super().__init__(models_bundle)
        self._density = (DensityDLinearForecaster(models_bundle)
                         if models_bundle.get('format') == 'density_dlinear' else None)

    def snapshot_options(self, inputs):
        if self._density is None:
            return super().snapshot_options(inputs)
        return {'density_history': pd.DataFrame(
            [point.model_dump() for point in inputs.density_observations], columns=['issue_time', 'n'])}

    def forecast(self, observations, *, issue_time=None, density_history=None, **kwargs):
        if self._density is None:
            return super().forecast(observations, issue_time=issue_time, **kwargs)
        if density_history is None:
            raise ValueError('Density DLinear requires unfilled density_history from Clio')
        return self.forecast_from_df(self._density.frame(issue_time or datetime.now(UTC), density_history))

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.features import build_features

        df = build_features(raw_observations_frame)
        return df
