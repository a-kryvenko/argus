"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['ThresholdForecastService', 'predict_revalidated_thresholds']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.threshold', name)
