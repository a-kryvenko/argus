"""Bounded background probes. Dashboard reads snapshots, never triggers jobs."""
import asyncio
import logging
import os
import ssl
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, unquote

import httpx
from fastapi.encoders import jsonable_encoder
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from common.schemas.forecast_release import PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastStatus
from app.db.models.monitoring import MonitorState
from app.db.session import get_session_factory

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


async def probe_http(client, name, base, path, host=None):
    if not base:
        return {'name': name, 'status': 'unknown', 'detail': 'Not configured'}
    started = time.perf_counter()
    try:
        response = await client.get(base.rstrip('/') + path, headers={'Host': host} if host else {})
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Unexpected response')
        return {'name': name, 'status': 'ok', 'response_ms': round((time.perf_counter()-started)*1000)}
    except (httpx.HTTPError, ValueError):
        return {'name': name, 'status': 'down', 'detail': 'Health check failed'}


async def redis_ping():
    url = os.getenv('MONITORING_REDIS_URL')
    if not url:
        return {'name': 'Redis', 'status': 'unknown', 'detail': 'Not configured'}
    writer = None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {'redis', 'rediss'} or not parsed.hostname:
            raise ValueError('Invalid Redis URL')
        async with asyncio.timeout(3):
            reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port or 6379,
                ssl=ssl.create_default_context() if parsed.scheme == 'rediss' else None)
            async def command(*parts):
                encoded = [p.encode() for p in parts]
                writer.write(b'*%d\r\n' % len(encoded) + b''.join(b'$%d\r\n' % len(p) + p + b'\r\n' for p in encoded))
                await writer.drain()
                return await reader.readline()
            if parsed.password:
                credentials = [unquote(parsed.password)]
                if parsed.username:
                    credentials.insert(0, unquote(parsed.username))
                if await command('AUTH', *credentials) != b'+OK\r\n':
                    raise ValueError('Authentication failed')
            if await command('PING') != b'+PONG\r\n':
                raise ValueError('Unexpected response')
        return {'name': 'Redis', 'status': 'ok'}
    except (OSError, ValueError, TimeoutError):
        return {'name': 'Redis', 'status': 'down', 'detail': 'Health check failed'}
    finally:
        if writer:
            writer.close()


def host_metrics(previous=None):
    proc = os.getenv('MONITORING_HOST_PROC')
    disk = os.getenv('MONITORING_DISK_PATH')
    result = {'status': 'unknown', 'cpu_percent': None, 'memory_percent': None,
              'disk_percent': None, 'disk_free_bytes': None}
    if not proc or not disk:
        return result
    try:
        values = [int(v) for v in (Path(proc)/'stat').read_text().splitlines()[0].split()[1:9]]
        total, idle = sum(values), values[3] + values[4]
        result['cpu_sample'] = [total, idle]
        old = (previous or {}).get('cpu_sample')
        if old and total > old[0] and idle >= old[1]:
            result['cpu_percent'] = round(max(0, min(100, 100*(1-(idle-old[1])/(total-old[0])))), 1)
        memory = {line.split(':')[0]: int(line.split()[1]) for line in (Path(proc)/'meminfo').read_text().splitlines()}
        result['memory_percent'] = round(100*(1-memory['MemAvailable']/memory['MemTotal']), 1)
        stat = os.statvfs(disk)
        result['disk_free_bytes'] = stat.f_bavail * stat.f_frsize
        result['disk_percent'] = round(100*(1-stat.f_bavail/stat.f_blocks), 1)
        result['status'] = 'warning' if result['disk_percent'] >= 90 or result['memory_percent'] >= 95 else 'ok'
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
        result['status'] = 'unknown'
    return result


async def observation_status(client):
    try:
        token = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
        if not token:
            raise ValueError('Not configured')
        response = await http_read(client, os.getenv('OBSERVATIONS_URL'), '/internal/v1/observations/monitoring', token)
        payload = response.json()
        data = payload['data']
        if payload.get('success') is not True or not isinstance(data.get('sources'), dict) or not isinstance(data.get('measurements'), list):
            raise ValueError('Invalid diagnostics')
        return data
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return {'status': 'unknown', 'sources': {}, 'measurements': [], 'error': 'Observation diagnostics unavailable'}


async def forecast_status(client, product):
    try:
        token = os.getenv('FORECASTS_SERVICE_TOKEN')
        if not token:
            raise ValueError('Not configured')
        response = await http_read(client, os.getenv('FORECASTS_URL'), f'/internal/v1/forecasts/{product}/status', token)
        data = ForecastStatus.model_validate(response.json())
        if data.product != product:
            raise ValueError('Unexpected product')
        result = data.model_dump(mode='json')
        # Keep internal diagnostics/exception text out of the browser payload.
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
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {'product': product, 'status': 'unknown', 'freshness': 'unavailable',
                'current_release': None, 'latest_attempt': None, 'latest_attempt_artifacts': [],
                'error': 'Forecast diagnostics unavailable'}


async def collect(previous=None):
    async with httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False,
                                 headers={'User-Agent': 'Argus-Monitor/1'}) as client:
        results = await asyncio.gather(
            probe_http(client, 'Website', os.getenv('MONITORING_SITE_URL'), '/', os.getenv('MONITORING_SITE_HOST')),
            probe_http(client, 'API', os.getenv('MONITORING_API_URL', 'http://127.0.0.1:8000'), '/ping'),
            probe_http(client, 'Clio', os.getenv('OBSERVATIONS_URL'), '/health/ready'),
            probe_http(client, 'Prophet', os.getenv('FORECASTS_URL'), '/health/ready'),
            redis_ping(), observation_status(client),
            *(forecast_status(client, p) for p in PRODUCT_ARTIFACTS),
        )
    # The caller has already queried the API database under the same transaction.
    services = results[:5] + [{'name': 'PostgreSQL (API)', 'status': 'ok'}]
    host = await asyncio.to_thread(host_metrics, (previous or {}).get('host'))
    observations, forecasts = results[5], results[6:]
    healthy = all(s['status'] == 'ok' for s in services) and observations['status'] == 'ok' and all(f['status'] == 'ok' for f in forecasts) and host['status'] == 'ok'
    return {'status': 'ok' if healthy else 'degraded', 'services': services,
            'observations': observations, 'forecasts': forecasts, 'host': host}


async def refresh():
    async with get_session_factory()() as db:
        await db.execute(text("SET LOCAL statement_timeout = '10s'"))
        if not await db.scalar(text('SELECT pg_try_advisory_xact_lock(730310)')):
            return
        previous = await db.get(MonitorState, 'project')
        now = datetime.now(UTC)
        if previous and now - previous.checked_at < timedelta(seconds=INTERVAL-1):
            return
        async with asyncio.timeout(12):
            payload = await collect(previous.payload if previous else None)
        await save_state(db, 'project', payload, datetime.now(UTC))
        await db.commit()


async def run_monitor_loop():
    while True:
        try:
            await refresh()
        except Exception:
            logger.warning('Project monitoring refresh failed', exc_info=True)
        await asyncio.sleep(INTERVAL)
