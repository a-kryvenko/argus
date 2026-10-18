"""Prophet selects the arrival model independently of collectors and speed heads."""
DEFAULT_MODEL = 'argus-cme-arrival-enlil-v1'


def forecast_events(inputs, *, registry=None):
    from forecast_core.solar_events.arrival import create_cme_arrival_model
    from forecast_core.solar_events.events import forecast as forecast_solar_wind_events
    entry = (registry or {}).get('cme_arrival', {})
    model = create_cme_arrival_model(entry.get('model', DEFAULT_MODEL))
    return forecast_solar_wind_events(inputs, arrival_model=model)
