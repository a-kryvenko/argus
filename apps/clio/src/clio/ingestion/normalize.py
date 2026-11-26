"""Prepare hourly model inputs from retained source measurements."""
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.normalized_observation import NormalizedObservation

from clio.ingestion.utils.products import PRODUCTS, WIND_METRICS
from clio.ingestion.utils.measurements import load_measurements
from clio.db.models.measurement import REQUIRED_METRICS, SOLAR_INDEX_METRICS, OBSERVATION_METRICS
from common.schemas.observation import Observation, ObservationPoint
from clio.db.session import get_db_session


NATIVE_WIND_PRODUCTS = ('swpc.rtsw_plasma', 'swpc.rtsw_mag')
DRIVER_TOLERANCE = {'kp': '2h', 'ap': '2h', 'dst': '0h', 'f10_7': '48h'}
OUTPUT_COLUMNS = ['observed_at', *OBSERVATION_METRICS, 'wind_time_basis']


def _wind_hours(frame, basis):
    if frame.empty:
        return pd.DataFrame(columns=[*WIND_METRICS, 'wind_time_basis'])
    hourly = (frame.assign(hour=frame.observed_at.dt.floor('h'))
              .groupby(['hour', 'metric']).value.mean().unstack('metric'))
    hourly = hourly.reindex(columns=WIND_METRICS).dropna()
    return hourly.assign(wind_time_basis=basis)


def _daily_driver(samples, hours, products, tolerance):
    values = pd.Series(np.nan, index=hours)
    days = pd.Series(pd.NaT, index=hours, dtype='datetime64[ns, UTC]')
    for product in products:
        current = samples.loc[samples.source_product.eq(product), ['observed_at', 'value']]
        if current.empty:
            continue
        current = current.assign(day=current.observed_at.dt.floor('D')).set_index('observed_at').sort_index()
        available = current.reindex(hours, method='ffill', tolerance=tolerance)
        use = available.day.notna() & (days.isna() | available.day.gt(days))
        values.loc[use] = available.loc[use, 'value']
        days.loc[use] = available.loc[use, 'day']
    return values


def normalize_measurements(measurements: pd.DataFrame, *, now=None, priorities=None) -> pd.DataFrame:
    """Assemble closed wind hours and use only previous or current driver values."""

    required_columns = {"metric", "value", "observed_at"}
    missing_columns = required_columns.difference(measurements.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Measurement frame is missing columns: {missing}")
    if measurements.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = measurements.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["metric"].isin(REQUIRED_METRICS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_at", "value"]
    )
    if 'quality' in frame:
        frame = frame.loc[frame.quality.isna()]
    if now is not None:
        closed_at = pd.Timestamp(now).floor('h')
        if 'received_at' in frame:
            receipts = pd.to_datetime(frame.received_at, utc=True)
            frame = frame.loc[receipts.isna() | (receipts <= now)]
        frame = frame.loc[frame.observed_at < closed_at]
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    wind_rows = frame.loc[frame.metric.isin(WIND_METRICS)]
    if 'source_product' in frame:
        native = _wind_hours(wind_rows.loc[wind_rows.source_product.isin(NATIVE_WIND_PRODUCTS)], 'l1')
        archive = _wind_hours(wind_rows.loc[wind_rows.source_product.eq('omni.hourly')], 'near_earth')
        archive = archive.loc[~archive.index.isin(native.index)]
        sources = [rows for rows in (native, archive) if not rows.empty]
        wind = pd.concat(sources) if sources else native
    else:
        wind = _wind_hours(wind_rows, 'l1')
    if wind.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    wind = wind.sort_index()
    for metric, duration in DRIVER_TOLERANCE.items():
        samples = frame.loc[frame.metric.eq(metric)]
        tolerance = pd.Timedelta(duration)
        if priorities and metric in priorities and 'source_product' in samples:
            ranks = {product: rank for rank, product in enumerate(priorities[metric])}
            samples = samples.loc[samples.source_product.isin(ranks)]
            if metric == 'f10_7':
                wind[metric] = _daily_driver(samples, wind.index, ranks, tolerance)
                continue
            samples = samples.assign(rank=samples.source_product.map(ranks)).sort_values('rank', kind='stable')
        series = samples.drop_duplicates('observed_at').set_index('observed_at').value.sort_index()
        wind[metric] = series.reindex(wind.index, method='ffill', tolerance=tolerance)
    wind = wind.dropna(subset=REQUIRED_METRICS)
    return wind.rename_axis('observed_at').reset_index().reindex(columns=OUTPUT_COLUMNS)


async def refresh_normalized_observations(
    config,
    now: datetime | None = None,
) -> Observation:
    """Write hourly rows from retained live measurements."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    
    since = now - timedelta(days=config.retention.normalized_days)
    priorities = {metric: [*policy.sources.live, *policy.sources.historical]
                  for metric, policy in config.observations.items() if metric in REQUIRED_METRICS}
    products = [name for name, product in PRODUCTS.items() if product.metrics.intersection(REQUIRED_METRICS)]
    async with get_db_session() as session:
        measurements = await load_measurements(
            session,
            since - timedelta(days=2),
            until=now + timedelta(microseconds=1),
            source_products=products,
            metrics=REQUIRED_METRICS,
        )
        normalized = normalize_measurements(measurements, now=now, priorities=priorities)
        normalized = normalized.loc[normalized.observed_at >= since]
        if 'source_product' in measurements:
            # Within native retention the raw series is authoritative. Removing
            # invalidated rows here allows an independently prepared archive to
            # fill the hour while keeping completed L1 history beyond retention.
            native = normalized.loc[normalized.wind_time_basis.eq('l1'), 'observed_at']
            await session.execute(delete(NormalizedObservation).where(
                NormalizedObservation.wind_time_basis == 'l1',
                NormalizedObservation.observed_at >= pd.Timestamp(now - timedelta(days=config.retention.live_days)).ceil('h'),
                NormalizedObservation.observed_at < pd.Timestamp(now).floor('h'),
                NormalizedObservation.observed_at.not_in(native.tolist()),
            ))
        if normalized.empty:
            if 'source_product' in measurements:
                await session.commit()
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
        record = {"observed_at": row.observed_at.to_pydatetime(), "wind_time_basis": row.wind_time_basis}
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
            set_={name: getattr(statement.excluded, name) for name in (*OBSERVATION_METRICS, 'wind_time_basis')},
            where=(statement.excluded.wind_time_basis == 'l1')
                  | NormalizedObservation.wind_time_basis.is_distinct_from('l1'),
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=[NormalizedObservation.observed_at])
    await session.execute(statement)
