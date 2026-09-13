"""Bounded, verified raw-hour cleanup. No aggregate deletion or recalculation."""
from datetime import UTC, datetime, timedelta
import logging
from sqlalchemy import delete, select, text
from sqlalchemy.exc import DBAPIError
from argus_clio.db.models import SolarWindObservation, SolarWindAggregate, SolarWindAggregatePending, SolarWindRetiredHour
from argus_clio.db.session import get_session_factory
from argus_clio.services.aggregation_audit import differences
from argus_clio.services.solar_wind_aggregation import VERSION, summarize

logger = logging.getLogger(__name__)


def verify_hour(raw, aggregates, kind, hour, now):
    saved = {(row.resolution_seconds, row.bucket_start): row for row in aggregates}
    for seconds in (300, 3600):
        for offset in range(0, 3600, seconds):
            start = hour+timedelta(seconds=offset)
            row = saved.get((seconds, start))
            if row is None:
                return {'reason': 'aggregate_missing', 'bucket_start': start, 'resolution_seconds': seconds}
            fields = differences(summarize(raw, kind, start, seconds, now), row.statistics)
            if row.version != VERSION:
                fields.append('version')
            if fields:
                return {'reason': 'aggregate_mismatch', 'bucket_start': start, 'resolution_seconds': seconds, 'fields': fields}
    return None


async def cleanup(*, apply=False, retention_days=90, limit=24, start=None, now=None):
    now = now or datetime.now(UTC)
    if retention_days < 90 or not 1 <= limit <= 240:
        raise ValueError('Retention must be at least 90 days; limit must be 1–240 source hours')
    cutoff = (now-timedelta(days=retention_days)).replace(minute=0, second=0, microsecond=0)
    if start is not None and (start.tzinfo is None or start != start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)):
        raise ValueError('from must be a whole UTC hour')
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(text('SET TRANSACTION READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '60s'"))
        candidates = list((await session.execute(text("""
            SELECT kind, date_trunc('hour', observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' AS hour
            FROM solar_wind_observation
            WHERE observed_at < :cutoff AND (CAST(:start AS timestamptz) IS NULL OR observed_at >= :start)
            GROUP BY 1, 2 ORDER BY 2, 1 LIMIT :limit
        """), {'cutoff': cutoff, 'start': start, 'limit': limit})).mappings())
    entries = []
    for candidate in candidates:
        kind, hour = candidate['kind'], candidate['hour']
        entry = {'kind': kind, 'hour': hour, 'rows': 0}
        try:
            async with factory() as session:
                if apply:
                    # A short table lock avoids relying on every raw writer using
                    # advisory locks. NOWAIT skips busy tables instead of blocking collection.
                    await session.execute(text('LOCK TABLE solar_wind_observation, solar_wind_aggregate_pending, solar_wind_aggregate IN SHARE ROW EXCLUSIVE MODE NOWAIT'))
                else:
                    await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
                await session.execute(text("SET LOCAL statement_timeout = '60s'"))
                if await session.get(SolarWindAggregatePending, (kind, hour)) is not None:
                    entries.append({**entry, 'status': 'skipped', 'reason': 'recalculation_pending'})
                    continue
                raw = list((await session.execute(select(SolarWindObservation).where(
                    SolarWindObservation.kind == kind, SolarWindObservation.observed_at >= hour,
                    SolarWindObservation.observed_at < hour+timedelta(hours=1)))).scalars())
                entry['rows'] = len(raw)
                if not raw:
                    entries.append({**entry, 'status': 'skipped', 'reason': 'raw_unavailable'})
                    continue
                aggregates = list((await session.execute(select(SolarWindAggregate).where(
                    SolarWindAggregate.kind == kind, SolarWindAggregate.bucket_start >= hour,
                    SolarWindAggregate.bucket_start < hour+timedelta(hours=1)))).scalars())
                problem = verify_hour(raw, aggregates, kind, hour, now)
                if problem:
                    entries.append({**entry, 'status': 'skipped', **problem})
                    continue
                if apply:
                    session.add(SolarWindRetiredHour(kind=kind, hour=hour, retired_at=now, raw_rows=len(raw)))
                    await session.flush()
                    result = await session.execute(delete(SolarWindObservation).where(
                        SolarWindObservation.kind == kind, SolarWindObservation.observed_at >= hour,
                        SolarWindObservation.observed_at < hour+timedelta(hours=1)))
                    if result.rowcount != len(raw):
                        raise RuntimeError('Raw row count changed during cleanup')
                    await session.commit()
                    logger.info('Retired %s %s: deleted %d raw rows', kind, hour.isoformat(), len(raw))
                entries.append({**entry, 'status': 'deleted' if apply else 'eligible'})
        except DBAPIError as exc:
            if getattr(exc.orig, 'sqlstate', None) != '55P03':
                raise
            entries.append({**entry, 'status': 'skipped', 'reason': 'database_busy'})
    return {'mode': 'apply' if apply else 'report', 'cutoff': cutoff, 'retention_days': retention_days,
            'limit': limit, 'examined_hours': len(entries),
            'eligible_rows': sum(e['rows'] for e in entries if e['status'] == 'eligible'),
            'deleted_rows': sum(e['rows'] for e in entries if e['status'] == 'deleted'),
            'skipped_hours': sum(e['status'] == 'skipped' for e in entries), 'hours': entries}
