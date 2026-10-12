"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['FORMAT', 'VERSION', 'FEATURES', 'logger', 'history_features', 'DensityProswinForecaster']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.density_proswin', name)
