"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['FORMAT', 'ridge_prediction', 'distribution', 'AIAWindForecaster', '__getattr__']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.aia_wind', name)
