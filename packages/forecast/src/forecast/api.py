"""Public solar-wind inference API, usable without any private backend."""
from forecast.calculation import ForecastResult, SnapshotService, calculate_forecast, calculate_snapshot
from forecast.inputs.observations import build_features
from forecast.inference.base import DefaultForecastService
from forecast.inference.quantiles import QuantileForecastService
from forecast.inference.threshold import ThresholdForecastService
from forecast.adapters.plasma import SWDensityFS, SWSpeedFS, SWSpeedProbaFS
from forecast.inference.rotation_dlinear import RotationDLinearForecaster
from forecast.inference.density_dlinear import DensityDLinearForecaster
from forecast.inference.temperature_proswin import TemperatureProswinForecaster
