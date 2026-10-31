"""Observation and forecast diagnostics. Infrastructure is monitored externally."""
import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta

import httpx
from fastapi.encoders import jsonable_encoder
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from common.schemas.forecast_release import PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastsStatus
from api.db.models.monitoring import MonitorState
from api.db.session import get_db_session

logger = logging.getLogger(__name__)
INTERVAL = 30
STALE_SECONDS = 120


async def save_state(db, name, payload, now):
    stmt = insert(MonitorState).values(name=name, payload=jsonable_encoder(payload), checked_at=now)
    await db.execute(stmt.on_conflict_do_update(index_elements=['name'],
        set_={'payload': stmt.excluded.payload, 'checked_at': now}))


async def http_read(client, base, path, token=None):
    if not base:
        raise ValueError('Not configured')
    response = await client.get(base.rstrip('/') + path,
        headers={'Authorization': f'Bearer {token}'} if token else {})
    response.raise_for_status()
    return response


async def observation_status(client):
    try:
        token = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
        if not token:
            raise ValueError('Not configured')
        response = await http_read(client, os.getenv('OBSERVATIONS_URL'), '/internal/v1/observations/status', token)
        payload = response.json()
        data = payload['data']
        if payload.get('success') is not True or not isinstance(data.get('sources'), dict) or not isinstance(data.get('measurements'), list):
            raise ValueError('Invalid diagnostics')
        return data
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return {'status': 'unknown', 'sources': {}, 'measurements': [], 'error': 'Observation diagnostics unavailable'}


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


async def collect():
    async with httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False,
                                 headers={'User-Agent': 'Argus-Monitor/1'}) as client:
        observations, forecasts = await asyncio.gather(
            observation_status(client),
            forecast_status(client),
        )
    return product_state({'observations': observations, 'forecasts': forecasts})


async def refresh():
    async with get_db_session() as db:
        await db.execute(text("SET LOCAL statement_timeout = '10s'"))
        if not await db.scalar(text('SELECT pg_try_advisory_xact_lock(730310)')):
            return
        previous = await db.get(MonitorState, 'project')
        now = datetime.now(UTC)
        if previous and now - previous.checked_at < timedelta(seconds=INTERVAL-1):
            return
        async with asyncio.timeout(12):
            payload = await collect()
        await save_state(db, 'project', payload, datetime.now(UTC))
        await db.commit()


async def run_monitor_loop():
    while True:
        try:
            await refresh()
        except Exception:
            logger.warning('Product diagnostics refresh failed', exc_info=True)
        await asyncio.sleep(INTERVAL)
