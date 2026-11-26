"""Read current observation and forecast status directly from their owners."""
import asyncio
import math
import os
from datetime import UTC, datetime, timedelta

import httpx
from common.config import get_config
from common.observationcatalog import METRICS
from common.schemas.forecast_release import PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastStatus

from api.services.upstream import UpstreamUnavailable, read_response


async def read_status(client: httpx.AsyncClient, path: str, url_env: str, token_env: str):
    url, token = os.getenv(url_env), os.getenv(token_env)
    if not url or not token:
        raise UpstreamUnavailable
    response = await read_response(
        client, 'GET', url, path, token, deadline=5, max_bytes=1024 * 1024, accepted=(200,),
    )
    if response.status_code != 200:
        raise UpstreamUnavailable
    return response.json()


def timestamp(value):
    if isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value)
            if stamp.tzinfo is not None:
                return stamp.astimezone(UTC)
        except (ValueError, OverflowError):
            pass
    return None


def status_from_observations(observations, *, now=None):
    now = now or datetime.now(UTC)
    policies = get_config().project_config['clio']['observations']
    measurements = []
    for metric, policy in policies.items():
        item = observations.get(metric)
        if not isinstance(item, dict):
            item = {}
        observed_at = timestamp(item.get('observed_at'))
        received = timestamp(item.get('received_at'))
        definition = METRICS.get(metric, {})
        interval = definition['native_seconds'] if definition.get('source_interval') else 0
        age = max(0, (now - observed_at).total_seconds() - interval) if observed_at is not None else None
        threshold = policy['freshness_minutes'] * 60
        state = ('unavailable' if observed_at is None or item.get('usable') is not True else
                 'future' if observed_at > now + timedelta(minutes=5) else 'delayed' if age > threshold else 'fresh')
        measurements.append(dict(metric=metric, latest_observation_at=observed_at, received_at=received,
                                 age_seconds=age, stale_after_seconds=threshold, status=state))
    return dict(generated_at=now, status='ok' if measurements and all(row['status'] == 'fresh' for row in measurements) else 'degraded',
                measurements=measurements)


async def observation_status(client):
    try:
        payload = await read_status(client, '/internal/v1/observations/status',
                                    'OBSERVATIONS_URL', 'OBSERVATIONS_SERVICE_TOKEN')
        if not isinstance(payload, dict) or payload.get('success') is not True or not isinstance(payload.get('data'), dict):
            raise ValueError('Invalid observation status')
        return status_from_observations(payload['data'])
    except (UpstreamUnavailable, ValueError, KeyError, TypeError, AttributeError):
        return {'status': 'unknown', 'measurements': [], 'error': 'Observation diagnostics unavailable'}


def forecast_diagnostic(data):
    result = data.model_dump(mode='json')
    result['note'] = ForecastStatus.model_fields['note'].default
    for key in ('current_release', 'latest_attempt'):
        if result.get(key):
            result[key].pop('input_diagnostics', None)
    if result.get('latest_attempt') and result['latest_attempt'].get('error'):
        result['latest_attempt']['error'] = 'Generation failed; see service logs'
    for artifact in result['latest_attempt_artifacts']:
        if artifact.get('error'):
            artifact['error'] = 'Artifact unavailable; see service logs'
    attempt = result['latest_attempt']
    failed = attempt and attempt['status'] in {'failed', 'partial', 'interrupted'}
    healthy = result['current_release'] is not None and result['freshness'] == 'within_age_limit' and not failed
    result['status'] = 'ok' if healthy else 'warning'
    return result


def forecast_diagnostics(payload):
    if not isinstance(payload, dict) or payload.get('contract_version', 1) != 1 or not isinstance(payload.get('products'), list):
        raise ValueError('Invalid forecast diagnostics')
    statuses = {}
    for item in payload['products']:
        product = item.get('product') if isinstance(item, dict) else None
        if not isinstance(product, str) or product not in PRODUCT_ARTIFACTS:
            continue
        if product in statuses:
            statuses[product] = None
            continue
        try:
            data = ForecastStatus.model_validate(item)
            ages = (data.release_age_hours, data.existing_public_max_age_hours)
            if any(age is not None and not math.isfinite(age) for age in ages):
                raise ValueError('Non-finite forecast age')
            statuses[product] = forecast_diagnostic(data)
        except ValueError:
            statuses[product] = None
    return statuses


async def forecast_status(client):
    try:
        payload = await read_status(client, '/internal/v1/forecasts/status', 'FORECASTS_URL', 'FORECASTS_SERVICE_TOKEN')
        statuses = forecast_diagnostics(payload)
    except (UpstreamUnavailable, ValueError):
        statuses = {}
    return [statuses.get(product) or {
        'product': product, 'status': 'unknown', 'freshness': 'unavailable',
        'current_release': None, 'latest_attempt': None, 'latest_attempt_artifacts': [],
        'error': 'Forecast diagnostics unavailable',
    } for product in PRODUCT_ARTIFACTS]


def product_state(payload):
    observations = payload.get('observations')
    forecasts = payload.get('forecasts', [])
    healthy = (observations or {}).get('status') == 'ok' and {
        item['product'] for item in forecasts if item.get('status') == 'ok'
    } == set(PRODUCT_ARTIFACTS)
    return {'status': 'ok' if healthy else 'degraded',
            'observations': observations, 'forecasts': forecasts}


async def read_project_status(client: httpx.AsyncClient):
    observations, forecasts = await asyncio.gather(observation_status(client), forecast_status(client))
    return {**product_state({'observations': observations, 'forecasts': forecasts}),
            'checked_at': datetime.now(UTC), 'stale': False}

