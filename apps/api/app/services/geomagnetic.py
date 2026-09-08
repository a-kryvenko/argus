import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.dataloaders.geomagnetic_loader import SOURCES, INTERVAL_SECONDS, POLL_SECONDS, fetch_records
from app.db.models import GeomagneticObservation
from app.db.session import get_session_factory
from app.services.collection_status import track_attempt
from app.services.history_coverage import coverage

logger = logging.getLogger(__name__)
# Allow the next native interval plus one hour of publication grace.
# Measure from interval END, not from start or polling time.
STALE_AFTER_SECONDS = {metric: seconds + 3600 for metric, seconds in INTERVAL_SECONDS.items()}


async def ingest_source(metric: str) -> None:
    async with get_session_factory()() as session:
        lock = await session.execute(text('SELECT pg_try_advisory_xact_lock(:key)'),
                                     {'key': 730200 if metric == 'kp' else 730201})
        if not lock.scalar_one():
            return
        async with track_attempt(metric, session, get_session_factory()) as attempt:
            records = await asyncio.to_thread(fetch_records, metric)
            # Replay the entire source window, including older revised intervals
            # and internal gaps left by downtime or late publication.
            attempt.received(records)
            statement = insert(GeomagneticObservation).values(records)
            statement = statement.on_conflict_do_update(
                index_elements=['metric', 'interval_start'],
                set_={key: statement.excluded[key] for key in ('interval_end', 'value', 'quality', 'received_at', 'raw')},
                where=GeomagneticObservation.raw.is_distinct_from(statement.excluded.raw),
            )
            await session.execute(statement)
        logger.info('%s: ingested %d intervals', metric, len(records))


async def refresh_geomagnetic() -> None:
    results = await asyncio.gather(*(ingest_source(metric) for metric in SOURCES), return_exceptions=True)
    failures = [metric for metric, result in zip(SOURCES, results) if isinstance(result, BaseException)]
    if failures:
        for metric, result in zip(SOURCES, results):
            if isinstance(result, BaseException):
                logger.error('%s collection failed: %s', metric, result)
        raise RuntimeError(f"Geomagnetic collection failed: {', '.join(failures)}")


def metadata(metric: str) -> dict:
    return {'label': 'Estimated Kp' if metric == 'kp' else 'Real-time Dst',
            'unit': '' if metric == 'kp' else 'nT',
            'source': 'NOAA SWPC' if metric == 'kp' else 'WDC Kyoto via NOAA SWPC',
            'source_url': SOURCES[metric], 'resolution_seconds': INTERVAL_SECONDS[metric],
            'poll_seconds': POLL_SECONDS[metric],
            'data_status': 'estimated' if metric == 'kp' else 'realtime',
            'stale_after_seconds': STALE_AFTER_SECONDS[metric], 'freshness_basis': 'interval_end',
            'time_basis': 'source_interval_start', 'gap_filling': 'none'}


def sample(record: GeomagneticObservation, now: datetime) -> dict:
    return {'interval_start': record.interval_start, 'interval_end': record.interval_end,
            'interval_status': 'in_progress' if record.interval_end > now else 'completed',
            'value': record.value, 'quality': record.quality, 'received_at': record.received_at,
            'station_count': record.raw.get('station_count')}


async def latest(session: AsyncSession, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    series = {}
    for metric in SOURCES:
        statement = (select(GeomagneticObservation)
                     .where(GeomagneticObservation.metric == metric, GeomagneticObservation.interval_start <= now)
                     .order_by(GeomagneticObservation.interval_start.desc()).limit(1))
        record = (await session.execute(statement)).scalars().first()
        lag = max(0, int((now-record.interval_end).total_seconds())) if record is not None else None
        series[metric] = {**metadata(metric), 'latest': sample(record, now) if record is not None else None,
                          'lag_seconds': lag,
                          'status': 'missing' if record is None or record.value is None else
                          'stale' if lag > STALE_AFTER_SECONDS[metric] else 'fresh'}
    return {'generated_at': now, 'series': series}


async def history(session: AsyncSession, start: datetime, end: datetime, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    # Include overlapping native intervals at the left edge, without clipping them.
    statement = (select(GeomagneticObservation)
                 .where(GeomagneticObservation.interval_start >= start - timedelta(hours=3),
                        GeomagneticObservation.interval_start < end,
                        GeomagneticObservation.interval_start <= now,
                        GeomagneticObservation.interval_end > start)
                 .order_by(GeomagneticObservation.interval_start))
    series = {metric: {**metadata(metric), 'points': []} for metric in SOURCES}
    for record in (await session.execute(statement)).scalars():
        series[record.metric]['points'].append(sample(record, now))
    for metric, item in series.items():
        item['coverage'] = coverage(item['points'], start, end, INTERVAL_SECONDS[metric], intervals=True, now=now)
    return {'from': start, 'to': end, 'selection': 'intervals_overlapping_[from,to)', 'series': series}
