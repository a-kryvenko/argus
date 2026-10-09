"""Public product adapters: snapshot conversion and algorithm selection."""
from forecast.inference.aia_wind import FORMAT
from common.adapters import observations_to_dataframe
from forecast.inference.density_dlinear import DensityDLinearForecaster
from datetime import UTC, datetime
import pandas as pd

from forecast.inference.quantiles import QuantileForecastService
from forecast.inference.threshold import ThresholdForecastService


class AIAWindServiceMixin:
    def snapshot_options(self, inputs):
        options = super().snapshot_options(inputs)
        if self.uses_aia:
            options['proswin_predictions'] = inputs.proswin_predictions
            options['proswin_ready_at'] = inputs.proswin_ready_at or inputs.as_of
            options['source_cutoff'] = inputs.as_of
            options['aia_features'] = pd.DataFrame([
                {**point.features, 'slot_at': point.slot_at,
                 'observed_at': point.observed_at, 'available_at': point.available_at}
                for point in inputs.aia_frames])
        return options

    def __init__(self,models_bundle):
        super().__init__(models_bundle)
        self.uses_aia=models_bundle.get('format')==FORMAT
        from forecast.inference.proswin_blend import ProswinBlendForecaster
        self._aia=ProswinBlendForecaster(models_bundle) if self.uses_aia else None
        if self.uses_aia:self.thresholds=models_bundle['thresholds']

    def forecast(self,observations,*,issue_time=None,speed_history=None,aia_features=None,proswin_predictions=(),proswin_ready_at=None,source_cutoff=None):
        if not self.uses_aia:return super().forecast(observations,issue_time=issue_time,speed_history=speed_history)
        issue_time=issue_time or datetime.now(UTC)
        history=observations_to_dataframe(observations) if speed_history is None else speed_history
        return self.forecast_from_df(self._aia.frame(issue_time,history,proswin_predictions, ready_at=proswin_ready_at, source_cutoff=source_cutoff))


class SWSpeedFS(AIAWindServiceMixin, QuantileForecastService):
    registry_name: str|None = "plasma_speed_quantile"
    target_name: str|None = "v"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.inputs.observations import build_features

        df = build_features(raw_observations_frame)
        return df

class SWSpeedProbaFS(AIAWindServiceMixin, ThresholdForecastService):
    registry_name: str|None = "plasma_speed_threshold"
    target_name: str|None = "v"

    def _build_features(self, raw_observations_frame: pd.DataFrame) -> pd.DataFrame:
        from forecast.inputs.observations import build_features

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
        from forecast.inputs.observations import build_features

        df = build_features(raw_observations_frame)
        return df
