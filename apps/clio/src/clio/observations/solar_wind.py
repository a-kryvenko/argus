"""Read models for unpropagated solar wind observations."""
from datetime import UTC, datetime, timedelta
import math

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from clio.providers.solar_wind_loader import SOURCES
from clio.db.models.measurement import Measurement
from clio.monitoring.specs import WIND_STALE_AFTER_SECONDS
from clio.observations.coverage import coverage

METADATA = {
    "bx": ("mag", "Bx", "nT", "GSM"),
    "by": ("mag", "By", "nT", "GSM"),
    "bz": ("mag", "Bz", "nT", "GSM"),
    "bt": ("mag", "Total magnetic field", "nT", None),
    "v": ("plasma", "Solar wind speed", "km/s", None),
    "n": ("plasma", "Proton density", "cm⁻³", None),
    "t": ("plasma", "Proton temperature", "K", None),
}


def metadata(metric: str) -> dict:
    kind, label, unit, coordinates = METADATA[metric]
    return {"label": label, "unit": unit, "coordinate_system": coordinates,
            "source": "NOAA SWPC RTSW", "source_url": SOURCES[kind],
            "location": "L1", "time_basis": "measurement", "propagated": False,
            "resolution_seconds": 60, "aggregation": "source_1_minute",
            "selection": "NOAA active spacecraft", "stale_after_seconds": WIND_STALE_AFTER_SECONDS}


def sample(record: Measurement, metric: str) -> dict:
    return {"observed_at": record.observed_at, "received_at": record.received_at,
            "value": record.value, "quality": record.quality}


async def latest(session: AsyncSession, metrics: list[str], now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    series = {}
    for metric in metrics:
        statement = (select(Measurement).where(Measurement.metric == metric,
                     Measurement.source_product == f'swpc.rtsw_{METADATA[metric][0]}',
                     Measurement.observed_at <= now).order_by(Measurement.observed_at.desc()).limit(1))
        record = (await session.execute(statement)).scalars().first()
        point = sample(record, metric) if record is not None else None
        age = max(0, int((now - record.observed_at).total_seconds())) if record is not None else None
        series[metric] = {**metadata(metric), "latest": point, "age_seconds": age,
                          "status": "missing" if point is None or point["value"] is None else
                          "stale" if age > WIND_STALE_AFTER_SECONDS else "fresh"}
    return {"generated_at": now, "series": series}


async def history(session: AsyncSession, metrics: list[str], start: datetime, end: datetime) -> dict:
    statement = (select(Measurement).where(Measurement.metric.in_(metrics),
                 Measurement.source_product.in_(['swpc.rtsw_mag', 'swpc.rtsw_plasma']),
                 Measurement.observed_at >= start, Measurement.observed_at < end)
                 .order_by(Measurement.observed_at))
    series = {metric: {**metadata(metric), "points": []} for metric in metrics}
    for record in (await session.execute(statement)).scalars():
        series[record.metric]['points'].append(sample(record, record.metric))
    for metric, item in series.items():
        item['coverage'] = coverage(item['points'], start, end, 60)
    return {"from": start, "to": end, "interval": "[from,to)", "gap_filling": "none", "series": series}


async def aggregated_history(session, metrics, start, end, seconds, now=None):
    if seconds not in (300, 3600):
        raise ValueError('Supported windows: 300 or 3600 seconds')
    now = now or datetime.now(UTC)
    left = datetime.fromtimestamp(math.floor(start.timestamp()/seconds)*seconds, UTC)
    right = datetime.fromtimestamp(math.floor(min(end, now).timestamp()/seconds)*seconds, UTC)
    # Deduplicate each minute before counting coverage. Only selected native
    # RTSW samples participate; historical hourly products have a different cadence.
    rows = (await session.execute(text("""
        WITH minutes AS (
            SELECT DISTINCT ON (metric, date_trunc('minute', observed_at))
                metric, observed_at, received_at, value, quality,
                to_timestamp(floor(extract(epoch FROM observed_at) / :seconds) * :seconds) AS bucket
            FROM measurement
            WHERE metric = ANY(:metrics)
              AND source_product IN ('swpc.rtsw_mag', 'swpc.rtsw_plasma')
              AND observed_at >= :start AND observed_at < :end
            ORDER BY metric, date_trunc('minute', observed_at), observed_at DESC
        ), samples AS (
            SELECT *, value IS NOT NULL AND quality IS DISTINCT FROM 'flagged'
                      AND quality IS DISTINCT FROM 'missing' AS usable
            FROM minutes
        )
        SELECT metric, bucket, max(received_at) AS received_at,
               count(*) AS samples, count(*) FILTER (WHERE usable) AS count,
               avg(value) FILTER (WHERE usable) AS mean,
               min(value) FILTER (WHERE usable) AS min,
               max(value) FILTER (WHERE usable) AS max,
               count(*) FILTER (WHERE usable AND value < 0) AS negative_count
        FROM samples GROUP BY metric, bucket ORDER BY bucket, metric
    """), {'metrics': metrics, 'seconds': seconds, 'start': left, 'end': right})).mappings().all()
    saved = {(row['metric'], row['bucket']): row for row in rows}
    series = {}
    for metric in metrics:
        points, gaps = [], []
        valid = invalid = missing = 0
        cursor = left
        while cursor < right:
            row = saved.get((metric, cursor))
            count = row['count'] if row else 0
            present = row['samples'] if row else 0
            valid += count
            invalid += present - count
            missing += seconds//60 - present
            if row and count:
                points.append(dict(observed_at=cursor, interval_end=cursor+timedelta(seconds=seconds),
                    received_at=row['received_at'], value=row['mean'], min=row['min'], max=row['max'],
                    count=count, expected_count=seconds//60, coverage_percent=round(100*count/(seconds//60), 2),
                    quality='unverified',
                    negative_count=row['negative_count'] if metric == 'bz' else None))
            if count < seconds//60:
                reason = 'partial' if present else 'missing'
                if gaps and gaps[-1]['to'] == cursor and gaps[-1]['reason'] == reason:
                    gaps[-1]['to'] = cursor+timedelta(seconds=seconds)
                    gaps[-1]['slots'] += seconds//60-count
                else:
                    gaps.append({'from': cursor, 'to': cursor+timedelta(seconds=seconds),
                                 'reason': reason, 'slots': seconds//60-count})
            cursor += timedelta(seconds=seconds)
        expected = valid+invalid+missing
        series[metric] = {**metadata(metric), 'resolution_seconds': seconds, 'aggregation': 'mean_min_max',
            'points': points, 'coverage': {'expected_slots': expected, 'usable_slots': valid,
            'missing_slots': missing, 'invalid_slots': invalid, 'percent': round(100*valid/expected, 2) if expected else None,
            'resolution_seconds': 60, 'evaluated_to': right, 'basis': 'sample_timestamps', 'gaps': gaps}}
    return {'from': start, 'to': end, 'evaluated_from': left, 'evaluated_to': right,
            'resolution_seconds': seconds,
            'selection': 'complete_UTC_buckets_starting_before_to_left_edge_may_extend_before_from',
            'gap_filling': 'none', 'series': series}
