"""The numeric Clio contract: last observations and bounded metric history."""
from datetime import UTC, timedelta
import math
from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AwareDatetime
from sqlalchemy import select, or_
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


def _points(rows, metrics):
    grouped = {}
    for time, metric, number in rows:
        grouped.setdefault(time, {'time': time})[metric] = number
    return [dict(time=time, **{metric: _output_value(metric, row.get(metric)) for metric in metrics})
            for time, row in sorted(grouped.items())]


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
                    NormalizedObservation.observed_at < end
                ).order_by(NormalizedObservation.observed_at).limit(MAX_ROWS + 1)
            records = (await session.execute(statement)).all()
            if len(records) > MAX_ROWS:
                raise ObservationUnavailable('Numeric read exceeds the supported batch size')
            for record in records:
                values = dict(zip(columns, record[1:]))
                rows.extend((record[0], metric, _value(metric, values[metric])) for metric in selected)
                if 'bt' in metrics:
                    components = [_value(metric, values[metric]) for metric in ('bx', 'by', 'bz')]
                    rows.append((record[0], 'bt', math.sqrt(sum(v*v for v in components))
                                 if all(v is not None for v in components) else None))
        native = [metric for metric in metrics if metric.endswith('_81mean')]
    else:
        native = metrics
    if native:
        statement = select(Measurement).where(
            Measurement.metric.in_(native), Measurement.observed_at >= start,
            Measurement.observed_at < end)
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
                if time < start or time + timedelta(minutes=5) > end:
                    continue
                # A minute contributes at most once, even for off-grid native samples.
                minute = record.observed_at.replace(second=0, microsecond=0)
                buckets.setdefault((time, record.metric), {})[minute] = _value(record.metric, record.value)
            for (time, metric), samples in buckets.items():
                valid = [v for v in samples.values() if v is not None]
                rows.append((time, metric, sum(valid) / len(valid) if valid else None))
        else:
            rows.extend((record.observed_at, record.metric, _value(record.metric, record.value))
                        for record in records)
    return {'interval': interval, 'points': _points(rows, metrics)}
