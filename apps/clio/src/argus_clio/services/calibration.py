"""Compatibility adapter for private derived observation preparation.

Provider parsing remains in the public clio library. Calibration, historical
solar estimates and the legacy model-input normalization remain private; this
adapter is used only by ingestion, never by the observation HTTP read path.
"""


def normalize_measurements(measurements):
    from forecast_core.api import normalize_measurements as normalize
    return normalize(measurements)


def load_solar_index_measurements(now):
    from forecast_core.api import load_solar_index_measurements as load
    return load(now)


def load_density_history(now):
    from forecast_core.api import load_density_history as load
    return load(now)


def merge_history(history, existing):
    from forecast_core.api import merge_history as merge
    return merge(history, existing)
