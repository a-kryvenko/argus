"""Compact numeric reads from hourly observations and native live measurements."""
from datetime import timedelta
import math

from sqlalchemy import select, func, or_

from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation
from clio.ingestion.products import OBSERVATIONS, PRODUCTS
from clio.observations.errors import ObservationUnavailable
from clio.observations.schema import OBSERVATION_METRICS, MEASUREMENT_METRICS

METRICS = (*MEASUREMENT_METRICS, 'bt')
LIVE_PRODUCTS = tuple(name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes)
SOLAR_ESTIMATES = {'s10', 'm10', 'y10'}
MAX_ROWS = 250_000


def value(metric, number, quality=None):
    if number is None or quality in ('flagged', 'missing') or metric in SOLAR_ESTIMATES:
        return None
    definition = OBSERVATIONS.get(metric)
    return float(number) if math.isfinite(number) and (definition.accepts(number) if definition else number >= 0) else None


def points(rows, metrics):
    grouped = {}
    for time, metric, number in rows:
        grouped.setdefault(time, {'time': time})[metric] = number
    return [dict(time=time, **{metric: row.get(metric) for metric in metrics})
            for time, row in sorted(grouped.items())]


async def latest(session, metrics, as_of):
    statement = (select(Measurement)
        .where(Measurement.metric.in_(metrics), Measurement.observed_at <= as_of,
               or_(Measurement.received_at.is_(None), Measurement.received_at <= as_of))
        .distinct(Measurement.metric).order_by(Measurement.metric, Measurement.observed_at.desc()))
    records = (await session.execute(statement)).scalars().all()
    # Each point contains only observations made at that time, not padded nulls
    # which would overwrite a different metric's last observation in consumers.
    grouped = {}
    for record in records:
        grouped.setdefault(record.observed_at, {'time': record.observed_at})[record.metric] = value(
            record.metric, record.value, record.quality)
    return {'points': [row for _, row in sorted(grouped.items())]}


async def history(session, metrics, start, end, interval, as_of):
    rows = []
    if interval == '1h':
        selected = [metric for metric in metrics if metric in OBSERVATION_METRICS]
        columns = list(selected)
        if 'bt' in metrics:
            columns = list(dict.fromkeys([*columns, 'bx', 'by', 'bz']))
        if columns:
            statement = select(NormalizedObservation.observed_at, *[
                getattr(NormalizedObservation, metric) for metric in columns]).where(
                    NormalizedObservation.observed_at >= start,
                    NormalizedObservation.observed_at < end,
                    NormalizedObservation.observed_at + timedelta(hours=1) <= as_of
                ).order_by(NormalizedObservation.observed_at).limit(MAX_ROWS + 1)
            records = (await session.execute(statement)).all()
            if len(records) > MAX_ROWS:
                raise ObservationUnavailable('Numeric read exceeds the supported batch size')
            for record in records:
                values = dict(zip(columns, record[1:]))
                rows.extend((record[0], metric, value(metric, values[metric])) for metric in selected)
                if 'bt' in metrics:
                    components = [value(metric, values[metric]) for metric in ('bx', 'by', 'bz')]
                    rows.append((record[0], 'bt', math.sqrt(sum(v*v for v in components))
                                 if all(v is not None for v in components) else None))
        native = [metric for metric in metrics if metric.endswith('_81mean')]
    else:
        native = metrics
    if native:
        statement = select(Measurement).where(
            Measurement.metric.in_(native), Measurement.observed_at >= start,
            Measurement.observed_at < end, Measurement.observed_at <= as_of,
            or_(Measurement.received_at.is_(None), Measurement.received_at <= as_of))
        if interval != '1h':
            statement = statement.where(or_(Measurement.source_product.in_(LIVE_PRODUCTS),
                                            Measurement.metric.in_([m for m in native if m.endswith('_81mean')])))
        records = (await session.execute(statement.order_by(Measurement.observed_at).limit(MAX_ROWS + 1))).scalars().all()
        if len(records) > MAX_ROWS:
            raise ObservationUnavailable('Numeric read exceeds the supported batch size')
        if interval == '5m':
            buckets = {}
            for record in records:
                time = record.observed_at.replace(minute=record.observed_at.minute // 5 * 5, second=0, microsecond=0)
                if time < start or time + timedelta(minutes=5) > min(end, as_of):
                    continue
                # A minute contributes at most once, even for off-grid native samples.
                minute = record.observed_at.replace(second=0, microsecond=0)
                buckets.setdefault((time, record.metric), {})[minute] = value(record.metric, record.value, record.quality)
            for (time, metric), samples in buckets.items():
                valid = [v for v in samples.values() if v is not None]
                rows.append((time, metric, sum(valid) / len(valid) if valid else None))
        else:
            rows.extend((record.observed_at, record.metric, value(record.metric, record.value, record.quality))
                        for record in records)
    return {'interval': interval, 'points': points(rows, metrics)}
