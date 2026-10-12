"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['KNOTS', 'WEIGHTS', 'VERSION', 'select_predictions', 'ProswinBlendForecaster']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.proswin_blend', name)
