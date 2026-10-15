"""Prophet selects the arrival model independently of collectors and speed heads."""
DEFAULT_MODEL = 'argus-cme-arrival-enlil-v1'


def forecast_events(inputs, *, registry=None, include_hss=True, include_dbem=True):
    from forecast_core.api import create_cme_arrival_model, forecast_solar_wind_events
    entry = (registry or {}).get('cme_arrival', {})
    model = create_cme_arrival_model(entry.get('model', DEFAULT_MODEL))
    return forecast_solar_wind_events(inputs, include_hss=include_hss, include_dbem=include_dbem,
                                      arrival_model=model)
