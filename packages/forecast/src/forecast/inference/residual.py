"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['ResidualRegressor']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inference.residual', name)
