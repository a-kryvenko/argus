"""Read and persist numeric source measurements."""
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.measurement import Measurement
from clio.observations.errors import ObservationUnavailable
from clio.observations.intervals import interval_end_expression
from clio.observations.schema import OBSERVATION_METRICS

UPSERT_BATCH_SIZE = 5_000


async def read_measurements(session, *, start, end, metrics, aggregation='raw',
                            min_value=None, max_value=None, exclude_value=None):
    timestamp = Measurement.observed_at
    available = func.greatest(timestamp, Measurement.received_at, interval_end_expression())
    if aggregation == 'hourly':
        timestamp = func.date_trunc('hour', timestamp)
        statement = select(Measurement.metric, func.avg(Measurement.value).label('value'),
                           timestamp.label('observed_at'), func.max(available).label('available_at'))
        statement = statement.group_by(Measurement.metric, timestamp)
    else:
        statement = select(Measurement.metric, Measurement.value, timestamp,
                           available.label('available_at'))
    statement = statement.where(
        Measurement.metric.in_(metrics), Measurement.observed_at >= start, Measurement.observed_at < end,
        or_(Measurement.quality.is_(None), Measurement.quality.not_in(['flagged', 'missing'])),
        Measurement.value > float('-inf'), Measurement.value < float('inf'),
    )
    if min_value is not None:
        statement = statement.where(Measurement.value >= min_value)
    if max_value is not None:
        statement = statement.where(Measurement.value < max_value)
    if exclude_value is not None:
        statement = statement.where(Measurement.value != exclude_value)
    rows = (await session.execute(statement.order_by(timestamp, Measurement.metric).limit(250_001))).all()
    if len(rows) > 250_000:
        raise ObservationUnavailable('Measurement range exceeds the supported batch size; request a smaller interval')
    return {'schema_version': 1, 'items': [dict(row._mapping) for row in rows]}


async def load_measurements(
    session: AsyncSession,
    since: datetime,
    until: datetime | None = None,
    source_products: list[str] | None = None,
) -> pd.DataFrame:
    statement = (select(Measurement.metric, Measurement.value, Measurement.observed_at)
                 .where(Measurement.observed_at >= since,
                        or_(Measurement.quality.is_(None), Measurement.quality.not_in(["flagged", "missing"])))
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
        else:
            statement = statement.on_conflict_do_nothing(constraint="uq_measurement_metric")
        await session.execute(statement)


async def store_native(session, records, *, source_priorities=None):
    # New polling timestamps alone do not revise an observation's receipt.
    fields = ('value', 'source_product', 'quality')
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
