import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from clio.providers.geomagnetic_loader import SOURCES, INTERVAL_SECONDS, fetch_records
from clio.db.models import Measurement
from clio.observations.native import geomagnetic_measurements, store_native
from clio.db.session import get_session_factory
from clio.db.locks import SOURCE_LOCKS
from clio.monitoring.specs import SOURCE_SPECS, poll_seconds
from clio.monitoring.status import track_attempt
from clio.coverage import coverage

logger = logging.getLogger(__name__)


async def ingest_source(metric: str) -> list[dict] | None:
    async with get_session_factory()() as session:
        lock = await session.execute(text('SELECT pg_try_advisory_xact_lock(:key)'),
                                     {'key': SOURCE_LOCKS[metric]})
        if not lock.scalar_one():
            return
        async with track_attempt(metric, session, get_session_factory()) as attempt:
            records = await asyncio.to_thread(fetch_records, metric)
            # Replay the entire source window, including older revised intervals
            # and internal gaps left by downtime or late publication.
            attempt.received(records)
            await store_native(session, geomagnetic_measurements(metric, records))
        logger.info('%s: ingested %d intervals', metric, len(records))
        return records


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
            'poll_seconds': poll_seconds(metric),
            'data_status': 'estimated' if metric == 'kp' else 'realtime',
            'stale_after_seconds': SOURCE_SPECS[metric]['stale_after_seconds'], 'freshness_basis': 'interval_end',
            'time_basis': 'source_interval_start', 'gap_filling': 'none'}


def sample(record: Measurement, now: datetime) -> dict:
    return {'interval_start': record.observed_at, 'interval_end': record.interval_end,
            'interval_status': 'in_progress' if record.interval_end > now else 'completed',
            'value': record.value, 'quality': record.quality, 'received_at': record.received_at,
            'station_count': record.station_count}


async def latest(session: AsyncSession, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    series = {}
    for metric in SOURCES:
        statement = (select(Measurement)
                     .where(Measurement.metric == metric, Measurement.interval_end.is_not(None), Measurement.observed_at <= now)
                     .order_by(Measurement.observed_at.desc()).limit(1))
        record = (await session.execute(statement)).scalars().first()
        lag = max(0, int((now-record.interval_end).total_seconds())) if record is not None else None
        series[metric] = {**metadata(metric), 'latest': sample(record, now) if record is not None else None,
                          'lag_seconds': lag,
                          'status': 'missing' if record is None or record.value is None else
                          'stale' if lag > SOURCE_SPECS[metric]['stale_after_seconds'] else 'fresh'}
    return {'generated_at': now, 'series': series}


async def history(session: AsyncSession, start: datetime, end: datetime, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    # Include overlapping native intervals at the left edge, without clipping them.
    statement = (select(Measurement)
                 .where(Measurement.metric.in_(list(SOURCES)),
                        Measurement.observed_at >= start - timedelta(hours=3),
                        Measurement.observed_at < end,
                        Measurement.observed_at <= now,
                        Measurement.interval_end > start)
                 .order_by(Measurement.observed_at))
    series = {metric: {**metadata(metric), 'points': []} for metric in SOURCES}
    for record in (await session.execute(statement)).scalars():
        series[record.metric]['points'].append(sample(record, now))
    for metric, item in series.items():
        item['coverage'] = coverage(item['points'], start, end, INTERVAL_SECONDS[metric], intervals=True, now=now)
    return {'from': start, 'to': end, 'selection': 'intervals_overlapping_[from,to)', 'series': series}


async def load_kp_measurements(session: AsyncSession, *, since: datetime, until: datetime):
    """Read Kp and Ap directly from their canonical measurements."""
    import pandas as pd
    statement = select(Measurement.metric, Measurement.value, Measurement.observed_at).where(
        Measurement.metric.in_(['kp', 'ap']), Measurement.value.is_not(None),
        Measurement.quality.not_in(['flagged', 'missing']),
        Measurement.observed_at >= since, Measurement.observed_at <= until,
        Measurement.received_at <= until).order_by(Measurement.observed_at, Measurement.metric)
    return pd.DataFrame((await session.execute(statement)).all(), columns=['metric', 'value', 'observed_at'])
