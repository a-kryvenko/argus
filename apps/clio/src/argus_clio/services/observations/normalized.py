from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from argus_clio.db.models import Measurement, NormalizedObservation
from common.schemas.observation import Observation, ObservationPoint
from clio.observations import (
    load_bootstrap_measurements, load_live_measurements,
    HISTORY_DAYS,
)
from argus_clio.services.observations.derived import (
    normalize_measurements,
    load_solar_index_measurements,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argus_clio.services.observations.store import (
    load_measurements, upsert_measurements, upsert_normalized_observations,
)


async def _database_is_empty(session: AsyncSession) -> bool:
    result = await session.execute(select(Measurement.id).limit(1))
    return result.scalar_one_or_none() is None


async def refresh_normalized_observations(
    session: AsyncSession,
    now: datetime | None = None,
) -> Observation:
    """Ingest raw source data, rebuild the wide layer, and return model input."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    if await _database_is_empty(session):
        bootstrap = await asyncio.to_thread(load_bootstrap_measurements, now)
        await upsert_measurements(session, bootstrap, track_receipt=False)

    live = await asyncio.to_thread(load_live_measurements)
    await upsert_measurements(session, live)
    solar = await asyncio.to_thread(load_solar_index_measurements, now)
    await upsert_measurements(session, solar)

    since = now - timedelta(days=HISTORY_DAYS)
    measurements = await load_measurements(session, since)
    normalized = normalize_measurements(measurements)
    if normalized.empty:
        raise RuntimeError("No complete normalized observations could be built")

    await upsert_normalized_observations(session, normalized)
    await session.commit()
    return await load_normalized_observations(session, since=since)


async def load_normalized_observations(
    session: AsyncSession,
    since: datetime | None = None,
    limit: int | None = None,
    until: datetime | None = None,
) -> Observation:
    statement = select(NormalizedObservation).order_by(
        NormalizedObservation.observed_at.desc()
    )
    if since is not None:
        statement = statement.where(NormalizedObservation.observed_at >= since)
    if until is not None:
        statement = statement.where(NormalizedObservation.observed_at <= until)
    if limit is not None:
        statement = statement.limit(limit)

    result = await session.execute(statement)
    records = list(reversed(result.scalars().all()))
    return Observation(points=[
        ObservationPoint(
            issue_time=record.observed_at,
            bx=record.bx,
            by=record.by,
            bz=record.bz,
            v=record.v,
            n=record.n,
            t=record.t,
            kp=int(record.kp),
            dst=int(record.dst),
            ap=int(record.ap),
            f10_7=int(record.f10_7),
            s10=record.s10,
            m10=record.m10,
            y10=record.y10,
        )
        for record in records
    ])
