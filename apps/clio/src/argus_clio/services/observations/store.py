"""Shared persistence operations; callers own transactions."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.observations import OBSERVATION_METRICS
from argus_clio.db.models import Measurement, NormalizedObservation, MeasurementReceipt

UPSERT_BATCH_SIZE = 5_000


async def upsert_measurements(
    session: AsyncSession,
    measurements: pd.DataFrame,
    *, track_receipt: bool = True, replace_existing: bool = True,
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
        if replace_existing:
            statement = statement.on_conflict_do_update(
                constraint="uq_measurement_metric",
                set_={"value": statement.excluded.value},
            )
        else:
            statement = statement.on_conflict_do_nothing(constraint="uq_measurement_metric")
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


async def load_measurements(
    session: AsyncSession,
    since: datetime,
    until: datetime | None = None,
) -> pd.DataFrame:
    statement = (select(Measurement.metric, Measurement.value, Measurement.observed_at)
                 .where(Measurement.observed_at >= since)
                 .order_by(Measurement.observed_at))
    if until is not None:
        statement = statement.where(Measurement.observed_at < until)
    result = await session.execute(statement)
    return pd.DataFrame(result.all(), columns=["metric", "value", "observed_at"])


async def upsert_normalized_observations(
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
