"""Public forecast contracts and lazy access to private model implementations."""
from forecast.calculation import ForecastResult, SnapshotService, calculate_forecast, calculate_snapshot
from forecast.inference.base import DefaultForecastService
from forecast._backend import resolve

_EXPORTS = {'build_features': ('forecast_core.solar_wind.inputs.observations', 'build_features'), 'QuantileForecastService': ('forecast_core.solar_wind.inference.quantiles', 'QuantileForecastService'), 'ThresholdForecastService': ('forecast_core.solar_wind.inference.threshold', 'ThresholdForecastService'), 'SWDensityFS': ('forecast_core.solar_wind.adapters.plasma', 'SWDensityFS'), 'SWSpeedFS': ('forecast_core.solar_wind.adapters.plasma', 'SWSpeedFS'), 'SWSpeedProbaFS': ('forecast_core.solar_wind.adapters.plasma', 'SWSpeedProbaFS'), 'RotationDLinearForecaster': ('forecast_core.solar_wind.inference.rotation_dlinear', 'RotationDLinearForecaster'), 'DensityDLinearForecaster': ('forecast_core.solar_wind.inference.density_dlinear', 'DensityDLinearForecaster'), 'TemperatureProswinForecaster': ('forecast_core.solar_wind.inference.temperature_proswin', 'TemperatureProswinForecaster')}

__all__ = ["ForecastResult", "SnapshotService", "calculate_forecast", "calculate_snapshot", "DefaultForecastService", *_EXPORTS]

def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = resolve(module, attribute)
    globals()[name] = value
    return value
