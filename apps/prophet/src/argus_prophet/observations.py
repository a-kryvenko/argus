"""Observation reads go exclusively through the owner's versioned HTTP API."""
import os
from datetime import UTC, datetime

import httpx
from common.schemas.forecast_inputs import ForecastInputs


def load_inputs(as_of: datetime | None = None) -> ForecastInputs:
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
        inputs = ForecastInputs.model_validate(response.json())
    if inputs.as_of != as_of:
        raise RuntimeError('Observation service returned a different as_of')
    if not inputs.observations.points:
        raise RuntimeError('No stored observations available in the last 30 days; check ingestion')
    return inputs
