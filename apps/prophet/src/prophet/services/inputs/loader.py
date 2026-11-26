"""Prepare and reuse each input source independently within a generation cycle."""
import asyncio
from datetime import UTC, datetime, timedelta
import logging
import os

import httpx

from common.schemas.observation import Observation
from common.schemas.solar_wind_events import EventInputs
from forecast.inputs import ForecastInputs
from prophet.services.generation.products import PRODUCTS
from prophet.services.inputs.observations import (
    apply_solar_measurements, read_native_history, read_observations, read_resource,
)

logger = logging.getLogger(__name__)


def read_source(source, as_of, *, inputs=None):
    """Run one source read or enrichment in a supervised child."""
    if source == 'proswin_cache':
        from prophet.services.proswin.cache import read_predictions
        return read_predictions(as_of)
    if source == 'proswin_queue':
        from prophet.services.proswin.queue import prepare
        prepared = prepare(inputs)
        return prepared.proswin_predictions, prepared.proswin_ready_at, prepared.proswin_job

    url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise RuntimeError('OBSERVATIONS_URL and OBSERVATIONS_SERVICE_TOKEN are required')
    timeout = 60 if source == 'events' else 180
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10), follow_redirects=False,
                      trust_env=False) as client:
        if source == 'hourly':
            return read_observations(client, url, token, as_of)
        if source == 'native_hourly':
            return read_native_history(client, url, token, as_of)
        if source == 'events':
            response = client.get(url.rstrip('/') + '/internal/v1/observations/solar-wind-events/inputs',
                                  headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
            return EventInputs.model_validate(response.json())
        if source == 'goes':
            from prophet.services.inputs.solar_indices import enrich_inputs
            files = read_resource(client.get, url, token, '/internal/v1/files', observation='goes',
                                  start=(as_of-timedelta(days=35)).isoformat(),
                                  end=(as_of+timedelta(microseconds=1)).isoformat())
            snapshot = ForecastInputs(as_of=as_of, read_at=datetime.now(UTC), observations=Observation(points=[]),
                                      files=files)
            enrich_inputs(snapshot)
            return snapshot.files, snapshot.measurements
    raise ValueError(f'Unknown input source: {source}')


async def load_product_inputs(product, as_of, cache, *, executor, timeout_seconds):
    spec = PRODUCTS[product]
    inputs = ForecastInputs(as_of=as_of, read_at=datetime.now(UTC), observations=Observation(points=[]))
    for source in (*spec.required_inputs, *spec.optional_inputs):
        if source == 'proswin_queue' and not inputs.speed_observations:
            continue
        if source not in cache:
            try:
                options = {'inputs': inputs} if source == 'proswin_queue' else {}
                cache[source] = await asyncio.to_thread(
                    executor, read_source, source, as_of, timeout_seconds=timeout_seconds, **options,
                )
            except Exception as exc:
                cache[source] = exc
        result = cache[source]
        if isinstance(result, Exception):
            if source in spec.required_inputs:
                raise RuntimeError(f'{source} input unavailable: {result}') from result
            logger.warning('%s optional input %s unavailable: %s', product, source, result)
            continue
        if source == 'hourly':
            inputs.read_at = result.read_at
            inputs.observations = result.observations.model_copy(deep=True)
            inputs.measurements = list(result.measurements)
            inputs.speed_observations = result.speed_observations
            inputs.density_observations = result.density_observations
            if spec.requires_normalized and not inputs.observations.points:
                raise RuntimeError('No stored observations available in the requested history range; check ingestion')
        elif source == 'native_hourly':
            inputs.solar_wind_hourly = result
        elif source == 'events':
            inputs.solar_wind_events = result
            if source in spec.required_inputs:
                inputs.read_at = result.read_at
        elif source == 'goes':
            inputs.files, calibrated = result
            apply_solar_measurements(inputs, calibrated)
        elif source == 'proswin_cache':
            inputs.proswin_predictions = result
        elif source == 'proswin_queue':
            inputs.proswin_predictions, inputs.proswin_ready_at, inputs.proswin_job = result
    return inputs
