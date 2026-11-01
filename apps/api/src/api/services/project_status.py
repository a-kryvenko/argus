"""Read current observation and forecast status directly from their owners."""
import asyncio
import os
from datetime import UTC, datetime, timedelta
import math
from common.config import get_config

import httpx
from common.schemas.forecast_release import PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastsStatus

async def http_read(client, base, path, token=None):
    if not base:
        raise ValueError('Not configured')
    response = await client.get(base.rstrip('/') + path,
        headers={'Authorization': f'Bearer {token}'} if token else {})
    response.raise_for_status()
    return response


def status_from_points(points, *, now=None):
    now = now or datetime.now(UTC)
    policies = get_config().project_config['clio']['observations']
    latest = {metric: (datetime.fromisoformat(row['time'].replace('Z', '+00:00')), value)
              for row in points for metric, value in row.items() if metric != 'time'}
    measurements = []
    for metric, policy in policies.items():
        if metric in ('goes', 'gong'):
            continue
        time, value = latest.get(metric, (None, None))
        duration = timedelta(hours=3 if metric in ('kp', 'ap') else 1 if metric == 'dst' else 0)
        age = max(0, (now-time-duration).total_seconds()) if time is not None else None
        threshold = policy['freshness_minutes'] * 60
        state = ('unavailable' if time is None or value is None or not math.isfinite(value) else
                 'future' if time > now+timedelta(minutes=5) else 'delayed' if age > threshold else 'fresh')
        measurements.append(dict(metric=metric, latest_observation_at=time, received_at=None,
                                 age_seconds=age, stale_after_seconds=threshold, status=state))
    return dict(generated_at=now, status='ok' if all(row['status'] == 'fresh' for row in measurements) else 'degraded',
                measurements=measurements)


async def observation_status(client):
    try:
        token = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
        if not token:
            raise ValueError('Not configured')
        response = await http_read(client, os.getenv('OBSERVATIONS_URL'), '/internal/v1/observations/latest', token)
        payload = response.json()
        if payload.get('success') is not True or not isinstance(payload['data'].get('points'), list):
            raise ValueError('Invalid observations')
        return status_from_points(payload['data']['points'])
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return {'status': 'unknown', 'measurements': [], 'error': 'Observation diagnostics unavailable'}


def forecast_diagnostic(data):
    result = data.model_dump(mode='json')
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
    result['status'] = 'ok' if result['freshness'] == 'within_age_limit' and not failed else 'warning'
    return result


async def forecast_status(client):
    try:
        token = os.getenv('FORECASTS_SERVICE_TOKEN')
        if not token:
            raise ValueError('Not configured')
        response = await http_read(client, os.getenv('FORECASTS_URL'), '/internal/v1/forecasts/status', token)
        data = ForecastsStatus.model_validate(response.json())
        return [forecast_diagnostic(item) for item in data.products]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return [{'product': product, 'status': 'unknown', 'freshness': 'unavailable',
                 'current_release': None, 'latest_attempt': None, 'latest_attempt_artifacts': [],
                 'error': 'Forecast diagnostics unavailable'} for product in PRODUCT_ARTIFACTS]


def product_state(payload):
    observations = payload.get('observations')
    forecasts = payload.get('forecasts', [])
    healthy = (observations or {}).get('status') == 'ok' and {
        item['product'] for item in forecasts if item.get('status') == 'ok'
    } == set(PRODUCT_ARTIFACTS)
    return {'status': 'ok' if healthy else 'degraded',
            'observations': observations, 'forecasts': forecasts}


async def read_project_status():
    async with httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False,
                                 headers={'User-Agent': 'Argus-Monitor/1'}) as client:
        observations, forecasts = await asyncio.gather(
            observation_status(client),
            forecast_status(client),
        )
    return {**product_state({'observations': observations, 'forecasts': forecasts}),
            'checked_at': datetime.now(UTC), 'stale': False}

