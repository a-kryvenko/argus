"""Build hourly observations from retained live measurements."""
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.normalized_observation import NormalizedObservation

from clio.ingestion.utils.products import PRODUCTS
from clio.ingestion.utils.measurements import load_measurements
from clio.db.models.measurement import REQUIRED_METRICS, SOLAR_INDEX_METRICS, OBSERVATION_METRICS
from common.schemas.observation import Observation, ObservationPoint
from clio.db.session import get_db_session


def normalize_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """Turn narrow source measurements into complete hourly observations."""

    required_columns = {"metric", "value", "observed_at"}
    missing_columns = required_columns.difference(measurements.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Measurement frame is missing columns: {missing}")
    if measurements.empty:
        return pd.DataFrame(columns=["observed_at", *OBSERVATION_METRICS])

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(REQUIRED_METRICS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )

    wide = frame.pivot_table(
        index="observed_at",
        columns="metric",
        values="value",
        aggfunc="last",
    )
    for metric in OBSERVATION_METRICS:
        if metric not in wide:
            wide[metric] = np.nan

    wide = wide[list(OBSERVATION_METRICS)].sort_index()
    required = wide[list(REQUIRED_METRICS)].dropna(how="all").resample("1h").first()
    required = required.interpolate(method="time", limit_area="inside")
    required = required.fillna(required.mean(numeric_only=True))
    required = required.dropna(subset=list(REQUIRED_METRICS))
    # Calibrated estimates are optional: never interpolate, backfill, or
    # propagate them into hours without a GOES-derived observation.
    solar = wide[list(SOLAR_INDEX_METRICS)].resample("1h").last()
    wide = required.join(solar)
    wide.columns.name = None
    return wide.reset_index()


async def refresh_normalized_observations(
    config,
    now: datetime | None = None,
) -> Observation:
    """Write hourly rows from retained live measurements."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    
    since = now - timedelta(days=config.retention.live_days)
    products = [name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes]
    async with get_db_session() as session:
        measurements = await load_measurements(
            session,
            since,
            until=now + timedelta(microseconds=1),
            source_products=products
        )
        normalized = normalize_measurements(measurements)
        if normalized.empty:
            raise RuntimeError("No complete normalized observations could be built; run collect first")

        await upsert_normalized_observations(session, normalized)
        await session.commit()
        result = await load_normalized_observations(session, since=since, until=now)
    
    return result


async def load_normalized_observations(
    session: AsyncSession, *, since: datetime, until: datetime,
) -> Observation:
    statement = select(NormalizedObservation).where(
        NormalizedObservation.observed_at >= since,
        NormalizedObservation.observed_at <= until,
    ).order_by(NormalizedObservation.observed_at)
    records = (await session.execute(statement)).scalars().all()
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
