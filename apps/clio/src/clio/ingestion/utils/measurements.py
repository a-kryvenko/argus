"""Read and persist numeric source measurements."""
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.measurement import Measurement
from clio.db.models.measurement import MEASUREMENT_METRICS

UPSERT_BATCH_SIZE = 5_000


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


async def upsert_measurements(
    session: AsyncSession,
    measurements: pd.DataFrame,
    *, replace_existing: bool = True,
    source_priorities: dict[str, list[str]] | None = None,
    repair_missing: bool = False,
) -> None:
    if measurements.empty:
        return

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(MEASUREMENT_METRICS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )
    frame = frame.drop_duplicates(["metric", "observed_at"], keep="last")
    records = [
        {
            "metric": row.metric,
            "value": float(row.value),
            "observed_at": row.observed_at.to_pydatetime(),
            "source_product": getattr(row, 'source_product', None),
            "received_at": getattr(row, 'received_at', None),
        }
        for row in frame.itertuples(index=False)
    ]

    for offset in range(0, len(records), UPSERT_BATCH_SIZE):
        statement = insert(Measurement).values(records[offset:offset + UPSERT_BATCH_SIZE])
        if replace_existing:
            changed = Measurement.value.is_distinct_from(statement.excluded.value)
            if source_priorities is not None:
                changed = and_(
                    or_(changed, Measurement.source_product.is_distinct_from(statement.excluded.source_product)),
                    or_(*(and_(Measurement.metric == metric, or_(
                        Measurement.source_product.is_(None),
                        Measurement.source_product.not_in(sources),
                        *(and_(statement.excluded.source_product == product,
                               Measurement.source_product.in_(sources[rank:]))
                          for rank, product in enumerate(sources)),
                    )) for metric, sources in source_priorities.items())),
                )
            statement = statement.on_conflict_do_update(
                constraint="uq_measurement_metric",
                set_={"value": statement.excluded.value,
                      "source_product": statement.excluded.source_product,
                      "received_at": statement.excluded.received_at},
                # Re-polling the same value must not erase known provenance.
                where=changed,
            )
        elif repair_missing:
            from clio.ingestion.utils.products import OBSERVATIONS
            nonnegative = [name for name, definition in OBSERVATIONS.items()
                           if definition.kind == 'numeric' and definition.signed is False]
            missing = or_(Measurement.value.is_(None), Measurement.value <= float('-inf'),
                          Measurement.value >= float('inf'),
                          and_(Measurement.metric.in_(nonnegative), Measurement.value < 0),
                          and_(Measurement.metric == 'kp', Measurement.value > 9))
            statement = statement.on_conflict_do_update(
                constraint='uq_measurement_metric',
                set_={'value': statement.excluded.value, 'source_product': statement.excluded.source_product,
                      'received_at': statement.excluded.received_at}, where=missing)
        else:
            statement = statement.on_conflict_do_nothing(constraint="uq_measurement_metric")
        await session.execute(statement)


async def store_native(session, records, *, source_priorities=None):
    # New polling timestamps alone do not revise an observation's receipt.
    fields = ('value', 'source_product')
    for offset in range(0, len(records), 1000):
        statement = insert(Measurement).values(records[offset:offset + 1000])
        changed = or_(*(getattr(Measurement, name).is_distinct_from(statement.excluded[name]) for name in fields))
        if source_priorities:
            allowed = or_(Measurement.metric.not_in(list(source_priorities)), *(and_(
                Measurement.metric == metric,
                or_(Measurement.source_product.is_(None), Measurement.source_product.not_in(sources),
                    *(and_(statement.excluded.source_product == product,
                           Measurement.source_product.in_(sources[rank:]))
                      for rank, product in enumerate(sources))),
            ) for metric, sources in source_priorities.items()))
            changed = and_(changed, allowed)
        await session.execute(statement.on_conflict_do_update(
            constraint='uq_measurement_metric',
            set_={name: statement.excluded[name] for name in (*fields, 'received_at')},
            where=changed))
