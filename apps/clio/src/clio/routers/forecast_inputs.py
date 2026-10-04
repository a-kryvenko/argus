"""Clio-owned read boundary for forecast workers."""
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime
from sqlalchemy import func, select, text, or_
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db import get_db_session
from clio.routers.auth import require_service_token
from clio.db.models import Measurement
from clio.observations.normalized import HISTORY_DAYS, load_normalized_observations
from common.schemas.forecast_inputs import DensityObservation, ObservationInputs, SourceMeasurement, SpeedObservation
from common.data.omni import OMNI_FILL_VALUES

from clio.routers.observation_files import load_observation_files
from clio.domains.solar_wind.history import history as solar_history

router = APIRouter(prefix='/internal/v1/observations', dependencies=[Depends(require_service_token)])


@router.get('/forecast-inputs', response_model=ObservationInputs)
async def forecast_inputs(as_of: AwareDatetime, session: AsyncSession = Depends(get_db_session)):
    as_of = as_of.astimezone(UTC)
    now = datetime.now(UTC)
    if as_of > now:
        raise HTTPException(422, 'as_of must not be in the future')
    # Both datasets see the same committed database state, even during ingestion.
    # read_at identifies read time, NOT a durable or replayable snapshot version.
    try:
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '60s'"))
        observations = await load_normalized_observations(
            session, since=as_of - timedelta(days=HISTORY_DAYS), until=as_of,
        )
        usable = or_(Measurement.quality.is_(None), Measurement.quality.not_in(['flagged', 'missing']))
        rows = (await session.execute(
            select(Measurement.metric, Measurement.value, Measurement.observed_at)
            .where(Measurement.metric.in_(['f10_7', 'dst', 'ap']), usable, Measurement.value.is_not(None),
                   Measurement.observed_at >= as_of - timedelta(days=88),
                   Measurement.observed_at <= as_of)
            .order_by(Measurement.observed_at, Measurement.metric)
            .limit(250_001)
        )).all()
        if len(rows) > 250_000:
            raise HTTPException(503, 'Forecast input range exceeds the supported batch size')
        # DLinear must receive observed speed, not the interpolated wide layer.
        hour = func.date_trunc('hour', Measurement.observed_at)
        speed_rows = (await session.execute(
            select(hour.label('issue_time'), func.avg(Measurement.value).label('v'))
            .where(Measurement.metric == 'v', usable,
                   Measurement.value >= 0,
                   Measurement.value < OMNI_FILL_VALUES['v'],
                   Measurement.observed_at >= as_of - timedelta(days=HISTORY_DAYS),
                   Measurement.observed_at <= as_of)
            .group_by(hour).order_by(hour)
        )).all()
        density_rows = (await session.execute(
            select(hour.label('issue_time'), func.avg(Measurement.value).label('n'))
            .where(Measurement.metric == 'n', usable,
                   Measurement.value >= 0,
                   Measurement.value != OMNI_FILL_VALUES['n'],
                   Measurement.value < float('inf'),
                   Measurement.observed_at >= as_of - timedelta(days=HISTORY_DAYS),
                   Measurement.observed_at <= as_of)
            .group_by(hour).order_by(hour)
        )).all()
        files = await load_observation_files(session, as_of)
        issue = as_of.replace(minute=0, second=0, microsecond=0)
        hourly = await solar_history(session, ['bx', 'by', 'bz', 'v', 'n', 't'],
                                     issue-timedelta(hours=168), issue, 3600, now=now)
        return ObservationInputs(
            as_of=as_of, read_at=now, observations=observations, files=files,
            solar_wind_hourly=hourly,
            measurements=[SourceMeasurement(metric=row.metric, value=row.value,
                                            observed_at=row.observed_at) for row in rows],
            speed_observations=[SpeedObservation(issue_time=row.issue_time, v=row.v) for row in speed_rows],
            density_observations=[DensityObservation(issue_time=row.issue_time, n=row.n) for row in density_rows],
        )
    finally:
        await session.rollback()
