"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['FORMAT', 'VERSION', 'HISTORY_NAMES', 'FEATURES', 'logger', 'aware', 'hourly_histories', 'TemperatureProswinForecaster']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.temperature_proswin', name)
