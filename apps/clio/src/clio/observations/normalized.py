"""Read and persist hourly observations."""
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.normalized_observation import NormalizedObservation
from clio.observations.schema import OBSERVATION_METRICS
from common.schemas.observation import Observation, ObservationPoint


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


async def upsert_normalized_observations(
    session: AsyncSession,
    observations: pd.DataFrame,
    *, replace_existing: bool = True,
) -> None:
    if observations.empty:
        return

    records = []
    for row in observations.itertuples(index=False):
        record = {"observed_at": row.observed_at.to_pydatetime()}
        record.update({
            metric: float(value) if pd.notna(value) else None
            for metric in OBSERVATION_METRICS
            for value in [getattr(row, metric)]
        })
        records.append(record)

    statement = insert(NormalizedObservation).values(records)
    if replace_existing:
        statement = statement.on_conflict_do_update(
            index_elements=[NormalizedObservation.observed_at],
            set_={metric: getattr(statement.excluded, metric) for metric in OBSERVATION_METRICS},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=[NormalizedObservation.observed_at])
    await session.execute(statement)
