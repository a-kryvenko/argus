"""Bounded in-process buffers, persisted hourly histogram buckets (30 days)."""
import asyncio
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from app.db.models.dashboard import ApiMetric
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)
BOUNDS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)
_buffer = {}


def record(route, method, status, duration_ms, *, user_agent=''):
    # Own health probes are operational checks, not API usage. Keep ordinary
    # /ping calls and other routes visible even when they use the same agent.
    if route == '/ping' and method == 'GET' and user_agent == 'Argus-Monitor/1':
        return
    # Route templates only, never arbitrary paths, query strings or identifiers.
    hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    bucket = next((b for b in BOUNDS if duration_ms <= b), -1)
    method = method if method in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD'} else 'OTHER'
    key = (hour, route[:256], method, status, bucket)
    if key not in _buffer and len(_buffer) >= 20000:
        return
    count, duration = _buffer.get(key, (0, 0.0))
    _buffer[key] = (count+1, duration+duration_ms)


async def flush():
    global _buffer
    pending, _buffer = _buffer, {}
    try:
        async with get_session_factory()() as db:
            for (hour, route, method, status, bucket), (count, duration) in pending.items():
                stmt = insert(ApiMetric).values(hour=hour, route=route, method=method, status=status,
                    bucket_ms=bucket, count=count, duration_ms=duration)
                await db.execute(stmt.on_conflict_do_update(index_elements=['hour', 'route', 'method', 'status', 'bucket_ms'],
                    set_={'count': ApiMetric.count + count, 'duration_ms': ApiMetric.duration_ms + duration}))
            await db.execute(delete(ApiMetric).where(ApiMetric.hour < datetime.now(timezone.utc)-timedelta(days=30)))
            await db.commit()
    except Exception:
        # Losing a metrics batch must never break the observation service.
        logger.warning('API statistics batch could not be persisted', exc_info=True)


async def run_flush_loop():
    while True:
        await asyncio.sleep(15)
        await flush()


def summary(rows):
    total = sum(r.count for r in rows)
    buckets = defaultdict(int)
    for row in rows:
        buckets[row.bucket_ms] += row.count
    cumulative = 0
    p95 = None
    for bound in sorted(buckets, key=lambda b: math.inf if b == -1 else b):
        cumulative += buckets[bound]
        if cumulative >= total * .95:
            p95 = bound
            break
    return {'requests': total, 'errors_4xx': sum(r.count for r in rows if 400 <= r.status < 500),
            'errors_5xx': sum(r.count for r in rows if r.status >= 500),
            'average_ms': round(sum(r.duration_ms for r in rows)/total, 2) if total else 0,
            'p95_upper_ms': p95}


def summarize(rows):
    routes, hours, statuses = defaultdict(list), defaultdict(list), defaultdict(int)
    for row in rows:
        routes[(row.route, row.method)].append(row)
        hours[row.hour.isoformat()].append(row)
        statuses[row.status] += row.count
    return {'summary': summary(rows),
        'routes': [{'route': route, 'method': method, **summary(items)} for (route, method), items in sorted(routes.items())],
        'hours': [{'hour': hour, **summary(items)} for hour, items in sorted(hours.items())],
        'statuses': [{'status': s, 'requests': c} for s, c in sorted(statuses.items())]}
