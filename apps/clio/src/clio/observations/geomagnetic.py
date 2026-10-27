from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from clio.providers.geomagnetic_loader import SOURCES, INTERVAL_SECONDS
from clio.db.models.measurement import Measurement
from clio.monitoring.specs import SOURCE_SPECS, poll_seconds
from clio.observations.coverage import coverage
from clio.observations.intervals import interval_end, interval_end_expression



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
    end = interval_end(record.metric, record.observed_at)
    return {'interval_start': record.observed_at, 'interval_end': end,
            'interval_status': 'in_progress' if end > now else 'completed',
            'value': record.value, 'quality': record.quality, 'received_at': record.received_at}


async def latest(session: AsyncSession, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    series = {}
    for metric in SOURCES:
        statement = (select(Measurement)
                     .where(Measurement.metric == metric, Measurement.observed_at <= now)
                     .order_by(Measurement.observed_at.desc()).limit(1))
        record = (await session.execute(statement)).scalars().first()
        lag = max(0, int((now-interval_end(record.metric, record.observed_at)).total_seconds())) if record is not None else None
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
                        interval_end_expression() > start)
                 .order_by(Measurement.observed_at))
    series = {metric: {**metadata(metric), 'points': []} for metric in SOURCES}
    for record in (await session.execute(statement)).scalars():
        series[record.metric]['points'].append(sample(record, now))
    for metric, item in series.items():
        item['coverage'] = coverage(item['points'], start, end, INTERVAL_SECONDS[metric], intervals=True, now=now)
    return {'from': start, 'to': end, 'selection': 'intervals_overlapping_[from,to)', 'series': series}
