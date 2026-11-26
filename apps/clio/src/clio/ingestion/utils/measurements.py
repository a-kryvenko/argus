"""Read and persist numeric source measurements."""
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.measurement import Measurement
from clio.ingestion.utils.products import OBSERVATIONS

UPSERT_BATCH_SIZE = 1_000


class WritePolicy(StrEnum):
    LIVE = 'live'
    REPAIR_INVALID = 'repair_invalid'


async def load_measurements(
    session: AsyncSession,
    since: datetime,
    until: datetime | None = None,
    source_products: list[str] | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    columns = ('metric', 'value', 'observed_at', 'source_product', 'received_at', 'quality', 'spacecraft')
    statement = (select(*(getattr(Measurement, name) for name in columns))
                 .where(Measurement.observed_at >= since)
                 .order_by(Measurement.observed_at))
    if until is not None:
        statement = statement.where(Measurement.observed_at < until)
    if source_products is not None:
        statement = statement.where(Measurement.source_product.in_(source_products))
    if metrics is not None:
        statement = statement.where(Measurement.metric.in_(metrics))
    result = await session.execute(statement)
    return pd.DataFrame(result.all(), columns=columns)


def historical_records(measurements: pd.DataFrame) -> list[dict]:
    if measurements.empty:
        return []

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(OBSERVATIONS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )
    if 'source_product' not in frame:
        frame['source_product'] = None
    frame = frame.drop_duplicates(["metric", "observed_at", "source_product"], keep="last")
    return [
        {
            "metric": row.metric,
            "value": float(row.value),
            "observed_at": row.observed_at.to_pydatetime(),
            **{
                name: None if pd.isna(value) else value
                for name in ('source_product', 'received_at', 'spacecraft')
                for value in (getattr(row, name, None),)
            },
        }
        for row in frame.itertuples(index=False)
    ]


def _live_update(excluded):
    # Identical polling responses preserve the receipt. Slower concurrent polls
    # cannot reverse a revision that has already committed.
    changed = or_(*(getattr(Measurement, name).is_distinct_from(excluded[name])
                    for name in ('value', 'quality', 'spacecraft')))
    return and_(changed, or_(Measurement.received_at.is_(None),
                             excluded.received_at >= Measurement.received_at))


def _invalid_stored_value():
    nonnegative = [name for name, definition in OBSERVATIONS.items()
                   if definition.kind == 'numeric' and not definition.signed]
    return or_(Measurement.value.is_(None), Measurement.value <= float('-inf'),
               Measurement.value >= float('inf'),
               and_(Measurement.metric.in_(nonnegative), Measurement.value < 0),
               and_(Measurement.metric == 'kp', Measurement.value > 9))


async def write_measurements(
    session: AsyncSession, records: list[dict], *, policy: WritePolicy,
) -> int:
    """Return the number of inserted or updated rows in the current transaction."""
    written = 0
    for offset in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = [{'quality': None, 'spacecraft': None, **row}
                 for row in records[offset:offset + UPSERT_BATCH_SIZE]]
        statement = insert(Measurement).values(batch)
        if policy == WritePolicy.LIVE:
            condition = _live_update(statement.excluded)
        elif policy == WritePolicy.REPAIR_INVALID:
            condition = and_(_invalid_stored_value(), or_(Measurement.received_at.is_(None),
                statement.excluded.received_at >= Measurement.received_at))
        else:
            raise ValueError(f'Unknown measurement write policy: {policy}')
        statement = statement.on_conflict_do_update(
            constraint='uq_measurement_metric',
            set_={name: statement.excluded[name] for name in ('value', 'quality', 'spacecraft', 'received_at')},
            where=condition,
        )
        result = await session.execute(statement.execution_options(preserve_rowcount=True))
        written += result.rowcount
    return written
