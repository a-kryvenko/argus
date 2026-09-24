"""Lazy access to private calibration computations; no provider IO."""


def load_solar_index_calibrations(path):
    from forecast_core.calibration import load_solar_index_calibrations as load
    return load(path)


def extract_solar_indices(goes, calibration):
    from forecast_core.calibration import extract_solar_indices as extract
    return extract(goes, calibration)


def extract_solar_index_observations(goes, calibrations, now):
    from forecast_core.calibration import extract_solar_index_observations as extract
    return extract(goes, calibrations, now)
