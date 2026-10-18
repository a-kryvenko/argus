"""Observation reads go exclusively through the owner's versioned HTTP API."""
import os
import logging
from datetime import UTC, datetime

import httpx
from common.schemas.forecast_inputs import ObservationInputs
from forecast.inputs import ForecastInputs


def load_inputs(as_of: datetime | None = None, *, prepare_proswin=False, include_events=False) -> ForecastInputs:
    url = os.getenv('OBSERVATIONS_URL')
    token = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise RuntimeError('OBSERVATIONS_URL and OBSERVATIONS_SERVICE_TOKEN are required')
    as_of = as_of or datetime.now(UTC)
    with httpx.Client(timeout=httpx.Timeout(180, connect=10), follow_redirects=False,
                      trust_env=False) as client:
        response = client.get(
            url.rstrip('/') + '/internal/v1/observations/forecast-inputs',
            params={'as_of': as_of.isoformat()},
            headers={'Authorization': f'Bearer {token}'},
        )
        response.raise_for_status()
        source = ObservationInputs.model_validate(response.json())
        inputs = ForecastInputs.model_validate(source.model_dump())
        if inputs.as_of != as_of:
            raise RuntimeError('Observation service returned a different as_of')
        if not inputs.observations.points:
            raise RuntimeError('No stored observations available in the requested history range; check ingestion')
        from prophet.services.inputs.solar_indices import enrich_inputs
        inputs = enrich_inputs(inputs, client, url, token)
        if include_events:
            try:
                inputs.solar_wind_events = read_event_inputs(client, url, token, as_of)
            except (httpx.HTTPError, ValueError, RuntimeError):
                logging.getLogger(__name__).exception('Event inputs unavailable; retaining background forecast')
            # Current event release is CME-only; no optional HSS image enrichment.
    from prophet.services.proswin.cache import read_predictions
    inputs.proswin_predictions = read_predictions(inputs.as_of)
    if prepare_proswin:
        from prophet.services.proswin.queue import prepare
        try:
            inputs = prepare(inputs)
        except (OSError, ValueError, RuntimeError):
            logging.getLogger(__name__).exception('PROSWIN queue unavailable; retaining causal cache / DLinear')
    return inputs


def read_event_inputs(client, url, token, as_of):
    from common.schemas.solar_wind_events import EventInputs
    response = client.get(url.rstrip('/') + '/internal/v1/observations/solar-wind-events/inputs',
                          params={'as_of': as_of.isoformat()},
                          headers={'Authorization': f'Bearer {token}'})
    response.raise_for_status()
    source = EventInputs.model_validate(response.json())
    if source.as_of != as_of:
        raise RuntimeError('Observation service returned a different event cutoff')
    return source


def load_event_inputs(as_of: datetime | None = None, *, include_speed=False):
    """Events tolerate missing normalized data and optional image/model inputs."""
    from common.schemas.observation import Observation
    url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise RuntimeError('OBSERVATIONS_URL and OBSERVATIONS_SERVICE_TOKEN are required')
    as_of = as_of or datetime.now(UTC)
    with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=False, trust_env=False) as client:
        source = read_event_inputs(client, url, token, as_of)
        inputs = ForecastInputs(as_of=as_of, read_at=source.read_at,
                                observations=Observation(points=[]), solar_wind_events=source)
        if include_speed:
            try:
                response = client.get(url.rstrip('/') + '/internal/v1/observations/forecast-inputs',
                    params={'as_of': as_of.isoformat()}, headers={'Authorization': f'Bearer {token}'})
                response.raise_for_status()
                context = ObservationInputs.model_validate(response.json())
                if context.as_of != as_of:
                    raise ValueError('Observation service returned a different speed cutoff')
                inputs = ForecastInputs.model_validate(context.model_dump())
                inputs.solar_wind_events = source
                from prophet.services.proswin.cache import read_predictions
                inputs.proswin_predictions = read_predictions(as_of)
            except (httpx.HTTPError, OSError, ValueError, RuntimeError):
                logging.getLogger(__name__).exception('Event speed context unavailable; retaining arrival forecasts')
        return inputs
