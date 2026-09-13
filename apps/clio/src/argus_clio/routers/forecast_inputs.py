"""Clio-owned read boundary for forecast workers."""
import os
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from argus_clio.db import get_db_session
from argus_clio.db.models import Measurement
from argus_clio.services.sensor_observations import load_normalized_observations
from common.schemas.forecast_inputs import DENSITY_METRICS, ForecastInputs, SourceMeasurement

security = HTTPBearer(auto_error=False)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not expected:
        raise HTTPException(503, 'Observation read service is not configured')
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, 'Invalid service credentials')


router = APIRouter(prefix='/internal/v1/observations', dependencies=[Depends(require_service_token)])


@router.get('/forecast-inputs', response_model=ForecastInputs)
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
            session, since=as_of - timedelta(days=30), until=as_of,
        )
        rows = (await session.execute(
            select(Measurement.metric, Measurement.value, Measurement.observed_at)
            .where(Measurement.metric.in_(DENSITY_METRICS),
                   Measurement.observed_at >= as_of - timedelta(days=88),
                   Measurement.observed_at <= as_of)
            .order_by(Measurement.observed_at, Measurement.metric)
            .limit(250_001)
        )).all()
        if len(rows) > 250_000:
            raise HTTPException(503, 'Forecast input range exceeds the supported batch size')
        return ForecastInputs(
            as_of=as_of, read_at=now, observations=observations,
            measurements=[SourceMeasurement(metric=row.metric, value=row.value,
                                            observed_at=row.observed_at) for row in rows],
        )
    finally:
        await session.rollback()
