"""Compatibility interface; implementation requires the private forecast backend."""
from forecast._backend import resolve

__all__ = ['HOUR', 'align']

def __getattr__(name):
    return resolve('forecast_core.solar_wind.inputs.aia.alignment', name)
