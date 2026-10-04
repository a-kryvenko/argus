from __future__ import annotations

from datetime import UTC, datetime, timedelta

from clio.db.models import NormalizedObservation
from common.schemas.observation import Observation, ObservationPoint
from clio.observations.schema import HISTORY_DAYS
from clio.observations.derived import normalize_measurements
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from clio.observations.store import (
    load_measurements, upsert_normalized_observations,
)


async def refresh_normalized_observations(
    session: AsyncSession,
    now: datetime | None = None,
) -> Observation:
    """Rebuild model inputs exclusively from persisted observations."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    since = now - timedelta(days=HISTORY_DAYS)
    measurements = await load_measurements(session, since, until=now + timedelta(microseconds=1))
    normalized = normalize_measurements(measurements)
    if normalized.empty:
        raise RuntimeError("No complete normalized observations could be built; run collect and backfill first")

    await upsert_normalized_observations(session, normalized)
    await session.commit()
    return await load_normalized_observations(session, since=since, until=now)


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
            s10=None,
            m10=None,
            y10=None,
        )
        for record in records
    ])
