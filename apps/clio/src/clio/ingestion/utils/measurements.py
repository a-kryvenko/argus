"""Read and persist numeric source measurements."""
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.measurement import MEASUREMENT_METRICS, Measurement
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
) -> pd.DataFrame:
    statement = (select(Measurement.metric, Measurement.value, Measurement.observed_at)
                 .where(Measurement.observed_at >= since)
                 .order_by(Measurement.observed_at))
    if until is not None:
        statement = statement.where(Measurement.observed_at < until)
    if source_products is not None:
        statement = statement.where(Measurement.source_product.in_(source_products))
    result = await session.execute(statement)
    return pd.DataFrame(result.all(), columns=["metric", "value", "observed_at"])


def historical_records(measurements: pd.DataFrame) -> list[dict]:
    if measurements.empty:
        return []

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(MEASUREMENT_METRICS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )
    frame = frame.drop_duplicates(["metric", "observed_at"], keep="last")
    return [
        {
            "metric": row.metric,
            "value": float(row.value),
            "observed_at": row.observed_at.to_pydatetime(),
            "source_product": getattr(row, 'source_product', None),
            "received_at": getattr(row, 'received_at', None),
        }
        for row in frame.itertuples(index=False)
    ]


def _live_update(excluded, source_priorities):
    # New polling timestamps alone do not revise an observation's receipt.
    changed = or_(Measurement.value.is_distinct_from(excluded.value),
                  Measurement.source_product.is_distinct_from(excluded.source_product))
    if not source_priorities:
        return changed

    allowed = [Measurement.metric.not_in(list(source_priorities))]
    for metric, sources in source_priorities.items():
        # The source can revise its own value or replace a lower-priority source.
        allowed_replacements = [
            and_(excluded.source_product == product, Measurement.source_product.in_(sources[rank:]))
            for rank, product in enumerate(sources)
        ]
        allowed.append(and_(Measurement.metric == metric, or_(
            Measurement.source_product.is_(None),
            Measurement.source_product.not_in(sources),
            *allowed_replacements,
        )))
    return and_(changed, or_(*allowed))


def _invalid_stored_value():
    nonnegative = [name for name, definition in OBSERVATIONS.items()
                   if definition.kind == 'numeric' and not definition.signed]
    return or_(Measurement.value.is_(None), Measurement.value <= float('-inf'),
               Measurement.value >= float('inf'),
               and_(Measurement.metric.in_(nonnegative), Measurement.value < 0),
               and_(Measurement.metric == 'kp', Measurement.value > 9))


async def write_measurements(
    session: AsyncSession, records: list[dict], *, policy: WritePolicy,
    source_priorities: dict[str, list[str]] | None = None,
) -> int:
    """Return the number of inserted or updated rows in the current transaction."""
    written = 0
    for offset in range(0, len(records), UPSERT_BATCH_SIZE):
        statement = insert(Measurement).values(records[offset:offset + UPSERT_BATCH_SIZE])
        if policy == WritePolicy.LIVE:
            condition = _live_update(statement.excluded, source_priorities)
        elif policy == WritePolicy.REPAIR_INVALID:
            condition = _invalid_stored_value()
        else:
            raise ValueError(f'Unknown measurement write policy: {policy}')
        statement = statement.on_conflict_do_update(
            constraint='uq_measurement_metric',
            set_={name: statement.excluded[name] for name in ('value', 'source_product', 'received_at')},
            where=condition,
        )
        result = await session.execute(statement.execution_options(preserve_rowcount=True))
        written += result.rowcount
    return written
