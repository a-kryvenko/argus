from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from argus_clio.db.models import Measurement, NormalizedObservation, MeasurementReceipt
from common.schemas.observation import Observation, ObservationPoint
from clio.observations import (
    load_bootstrap_measurements as _load_bootstrap_measurements, load_live_measurements,
    OBSERVATION_METRICS, REQUIRED_METRICS, SOLAR_INDEX_METRICS,
    HISTORY_DAYS,
)
from argus_clio.services.derived_observations import (
    normalize_measurements,
    load_solar_index_measurements as _load_solar_index_measurements,

)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

UPSERT_BATCH_SIZE = 5_000
logger = logging.getLogger(__name__)


async def _database_is_empty(session: AsyncSession) -> bool:
    result = await session.execute(select(Measurement.id).limit(1))
    return result.scalar_one_or_none() is None


async def _upsert_measurements(
    session: AsyncSession,
    measurements: pd.DataFrame,
    *, track_receipt: bool = True,
) -> None:
    if measurements.empty:
        return

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(OBSERVATION_METRICS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )
    frame = frame.drop_duplicates(["metric", "observed_at"], keep="last")
    records = [
        {
            "metric": row.metric,
            "value": float(row.value),
            "observed_at": row.observed_at.to_pydatetime(),
        }
        for row in frame.itertuples(index=False)
    ]

    for offset in range(0, len(records), UPSERT_BATCH_SIZE):
        statement = insert(Measurement).values(records[offset:offset + UPSERT_BATCH_SIZE])
        statement = statement.on_conflict_do_update(
            constraint="uq_measurement_metric",
            set_={"value": statement.excluded.value},
        )
        await session.execute(statement)

    # Metadata and observations commit together; older backfills cannot replace
    # the receipt timestamp of a newer observation.
    if not track_receipt:
        return
    received = datetime.now(UTC)
    for metric, group in frame.groupby('metric'):
        latest = group['observed_at'].max().to_pydatetime()
        stmt = insert(MeasurementReceipt).values(metric=metric, latest_observation_at=latest, received_at=received)
        await session.execute(stmt.on_conflict_do_update(index_elements=['metric'],
            set_={'latest_observation_at': latest, 'received_at': received},
            where=MeasurementReceipt.latest_observation_at <= latest))


async def _load_measurements(
    session: AsyncSession,
    since: datetime,
) -> pd.DataFrame:
    result = await session.execute(
        select(Measurement.metric, Measurement.value, Measurement.observed_at)
        .where(Measurement.observed_at >= since)
        .order_by(Measurement.observed_at)
    )
    return pd.DataFrame(result.all(), columns=["metric", "value", "observed_at"])


async def _upsert_normalized_observations(
    session: AsyncSession,
    observations: pd.DataFrame,
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
    statement = statement.on_conflict_do_update(
        index_elements=[NormalizedObservation.observed_at],
        set_={
            metric: getattr(statement.excluded, metric)
            for metric in OBSERVATION_METRICS
        },
    )
    await session.execute(statement)


async def refresh_normalized_observations(
    session: AsyncSession,
    now: datetime | None = None,
) -> Observation:
    """Ingest raw source data, rebuild the wide layer, and return model input."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    if await _database_is_empty(session):
        bootstrap = await asyncio.to_thread(_load_bootstrap_measurements, now)
        await _upsert_measurements(session, bootstrap, track_receipt=False)

    live = await asyncio.to_thread(load_live_measurements)
    await _upsert_measurements(session, live)
    solar = await asyncio.to_thread(_load_solar_index_measurements, now)
    await _upsert_measurements(session, solar)

    since = now - timedelta(days=HISTORY_DAYS)
    measurements = await _load_measurements(session, since)
    normalized = normalize_measurements(measurements)
    if normalized.empty:
        raise RuntimeError("No complete normalized observations could be built")

    await _upsert_normalized_observations(session, normalized)
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
