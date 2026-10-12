"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['SOLAR_ROTATION_DAYS', 'build_features']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inputs.observations', name)
