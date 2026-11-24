"""The numeric Clio contract: last observations and bounded metric history."""
from datetime import UTC, timedelta
import math
from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AwareDatetime
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation
from clio.ingestion.utils.products import OBSERVATIONS, PRODUCTS
from clio.db.models.measurement import OBSERVATION_METRICS, MEASUREMENT_METRICS
from clio.routers.errors import ObservationUnavailable

METRICS = (*MEASUREMENT_METRICS, 'bt')
LIVE_PRODUCTS = tuple(name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes)
SOLAR_ESTIMATES = {'s10', 'm10', 'y10'}
INTEGER_METRICS = {'kp', 'dst', 'ap', 'f10_7'}
MAX_ROWS = 250_000

router = APIRouter(prefix='/internal/v1/observations', tags=['observations'])


def selected_metrics(metrics: str | None = None):
    selected = list(METRICS) if metrics is None else list(dict.fromkeys(m.strip() for m in metrics.split(',')))
    if not selected or any(m not in METRICS for m in selected):
        raise HTTPException(422, 'Unknown observation metric')
    return selected


@router.get('/latest')
async def latest(response: Response, metrics: list[str] = Depends(selected_metrics),
                 session: AsyncSession = Depends(get_request_db_session)):
    response.headers['Cache-Control'] = 'no-store'
    try:
        return success_response(await _read_latest(session, metrics))
    except SQLAlchemyError:
        raise HTTPException(503, 'Numeric observation storage is unavailable') from None


@router.get('/history')
async def history(response: Response, start: AwareDatetime = Query(alias='from'),
                  end: AwareDatetime = Query(alias='to'),
                  interval: Literal['1m', '5m', '1h'] = '1h',
                  metrics: list[str] = Depends(selected_metrics),
                  session: AsyncSession = Depends(get_request_db_session)):
    if not timedelta(0) < end-start <= timedelta(days=366):
        raise HTTPException(422, 'Use an interval of at most 366 days')
    response.headers['Cache-Control'] = 'no-store'
    try:
        data = await _read_history(session, metrics, start.astimezone(UTC), end.astimezone(UTC), interval)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except SQLAlchemyError:
        raise HTTPException(503, 'Numeric observation storage is unavailable') from None
    return success_response(data)


def _value(metric, number):
    if number is None or metric in SOLAR_ESTIMATES:
        return None
    definition = OBSERVATIONS.get(metric)
    return float(number) if (definition.accepts(number) if definition else math.isfinite(number) and number >= 0) else None


def _output_value(metric, number):
    """Apply the integer observation contract after validation and aggregation."""
    return int(number) if metric in INTEGER_METRICS and number is not None else number


async def _read_latest(session, metrics):
    statement = (select(Measurement)
        .where(Measurement.metric.in_(metrics))
        .distinct(Measurement.metric).order_by(Measurement.metric, Measurement.observed_at.desc()))
    records = (await session.execute(statement)).scalars().all()
    # Each point contains only observations made at that time, not padded nulls
    # which would overwrite a different metric's last observation in consumers.
    grouped = {}
    for record in records:
        grouped.setdefault(record.observed_at, {'time': record.observed_at})[record.metric] = _output_value(
            record.metric, _value(record.metric, record.value))
    return {'points': [row for _, row in sorted(grouped.items())]}


async def _read_history(session, metrics, start, end, interval):
    hourly = interval == '1h'
    means = [metric for metric in metrics if metric.endswith('_81mean')]
    points = await _read_hourly_points(session, metrics, start, end) if hourly else {}
    native_metrics = means if hourly else metrics
    if native_metrics:
        statement = select(Measurement.observed_at, Measurement.metric, Measurement.value).where(
            Measurement.metric.in_(native_metrics))
        if not hourly:
            statement = statement.where(or_(Measurement.source_product.in_(LIVE_PRODUCTS), Measurement.metric.in_(means)))
        records = await _read_bounded_rows(session, statement, Measurement.observed_at, start, end)
        if interval == '5m':
            points = _five_minute_points(records, start, end)
        else:
            for record in records:
                points.setdefault(record.observed_at, {})[record.metric] = _value(record.metric, record.value)

    return {
        'interval': interval,
        'points': [dict(time=time, **{metric: _output_value(metric, values.get(metric)) for metric in metrics})
                   for time, values in sorted(points.items())],
    }


async def _read_bounded_rows(session, statement, observed_at, start, end):
    statement = statement.where(observed_at >= start, observed_at < end).order_by(observed_at).limit(MAX_ROWS + 1)
    rows = (await session.execute(statement)).all()
    if len(rows) > MAX_ROWS:
        raise ObservationUnavailable('Numeric read exceeds the supported batch size')
    return rows


async def _read_hourly_points(session, metrics, start, end):
    components = ('bx', 'by', 'bz') if 'bt' in metrics else ()
    selected = [metric for metric in metrics if metric in OBSERVATION_METRICS]
    columns = tuple(dict.fromkeys((*selected, *components)))
    if not columns:
        return {}

    observed_at = NormalizedObservation.observed_at
    statement = select(observed_at, *(getattr(NormalizedObservation, metric) for metric in columns))
    rows = await _read_bounded_rows(session, statement, observed_at, start, end)
    points = {}
    for time, *numbers in rows:
        values = {metric: _value(metric, number) for metric, number in zip(columns, numbers)}
        if components:
            field = [values[metric] for metric in components]
            values['bt'] = math.sqrt(sum(value * value for value in field)) if None not in field else None
        points[time] = values
    return points


def _five_minute_points(records, start, end):
    window = timedelta(minutes=5)
    buckets = {}
    for record in records:
        minute = record.observed_at.replace(second=0, microsecond=0)
        time = minute.replace(minute=minute.minute // 5 * 5)
        if start <= time and time + window <= end:
            # In timestamp order, the last sample within each minute wins.
            buckets.setdefault((time, record.metric), {})[minute] = _value(record.metric, record.value)

    points = {}
    for (time, metric), samples in buckets.items():
        valid = [value for value in samples.values() if value is not None]
        points.setdefault(time, {})[metric] = sum(valid) / len(valid) if valid else None
    return points
