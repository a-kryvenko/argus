"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['QUANTILE_NAMES', 'uses_overlapping_buckets', 'validate_overlapping_buckets', 'overlapping_bucket_weights', 'predict_overlapping_quantiles', 'apply_quantile_calibration', 'QuantileForecastService']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.quantiles', name)
