from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from app.db.models import Measurement, NormalizedObservation
from clio.dataloaders.spdf_loader import SPDF_Loader
from clio.dataloaders.swpc_loader import SWPC_Loader
from clio.dataloaders.goes_loader import fetch_goes
from common.config import get_config
from common.schemas.observation import Observation, ObservationPoint
from forecast.data_pipelines.solar_indices import (
    extract_solar_index_observations,
    load_solar_index_calibrations,
)
from requests import RequestException
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

REQUIRED_METRICS = SWPC_Loader.METRICS
SOLAR_INDEX_METRICS = ("s10", "m10", "y10")
OBSERVATION_METRICS = (*REQUIRED_METRICS, *SOLAR_INDEX_METRICS)
HISTORY_DAYS = 30
LIVE_SOURCE_DAYS = 6
UPSERT_BATCH_SIZE = 5_000
logger = logging.getLogger(__name__)


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
    frame = frame[frame["metric"].isin(OBSERVATION_METRICS)]
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


def _wide_to_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = "issue_time" if "issue_time" in frame else "observed_at"
    metrics = [metric for metric in OBSERVATION_METRICS if metric in frame]
    if timestamp_column not in frame or not metrics:
        return pd.DataFrame(columns=["metric", "value", "observed_at"])

    return (
        frame
        .melt(
            id_vars=timestamp_column,
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
        .rename(columns={timestamp_column: "observed_at"})
        .dropna(subset=["observed_at", "value"])
    )


def _load_bootstrap_measurements(now: datetime) -> pd.DataFrame:
    config = get_config()
    legacy_path = config.project_config.get("paths", {}).get("live_sensors")
    if legacy_path:
        legacy_path = config.workdir / Path(legacy_path)
        if legacy_path.is_file():
            legacy = pd.read_csv(legacy_path, parse_dates=["issue_time"])
            measurements = _wide_to_measurements(legacy)
            if not measurements.empty:
                return measurements

    historical = SPDF_Loader.load(
        start_date=now - timedelta(days=HISTORY_DAYS),
        end_date=now - timedelta(days=LIVE_SOURCE_DAYS - 1),
    )
    return _wide_to_measurements(historical)


def _load_solar_index_measurements(now: datetime) -> pd.DataFrame:
    """Use the configured frozen calibration; failures leave indices absent."""
    empty = pd.DataFrame(columns=["metric", "value", "observed_at"])
    config = get_config()
    registry = config.models_registry.get("models", {}).get("solar_index_calibration", {})
    path = registry.get("calibration_path")
    if not path:
        logger.warning("Solar-index observations unavailable: no calibration configured")
        return empty
    try:
        calibrations = load_solar_index_calibrations(config.workdir / Path(path))
        goes = fetch_goes(end=now)
        estimates = extract_solar_index_observations(goes, calibrations, pd.Timestamp(now))
    except FileNotFoundError as exc:
        logger.warning(
            "Solar-index calibration file not found: %s. S10/M10/Y10 remain "
            "unavailable; other observations continue updating. Train a calibration "
            "with forecast.data_pipelines.calibrate_solar_indices or set "
            "models.solar_index_calibration.calibration_path in models_registry.yaml "
            "to an existing artifact.",
            exc.filename,
        )
        return empty
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, RequestException):
        logger.exception("Solar-index observations unavailable: calibration or GOES failure")
        return empty

    # The live feed is a rolling day. Only persist the current hour snapshot,
    # so a later truncated feed cannot revise earlier daily estimates.
    estimates = estimates.loc[estimates["observed_at"] == pd.Timestamp(now).floor("h")]
    if estimates.empty:
        logger.warning("Solar-index observations unavailable: no recent valid GOES samples")
        return empty
    return _wide_to_measurements(estimates)


async def _database_is_empty(session: AsyncSession) -> bool:
    result = await session.execute(select(Measurement.id).limit(1))
    return result.scalar_one_or_none() is None


async def _upsert_measurements(
    session: AsyncSession,
    measurements: pd.DataFrame,
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
        await _upsert_measurements(session, bootstrap)

    live = await asyncio.to_thread(SWPC_Loader.load_measurements)
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
) -> Observation:
    statement = select(NormalizedObservation).order_by(
        NormalizedObservation.observed_at.desc()
    )
    if since is not None:
        statement = statement.where(NormalizedObservation.observed_at >= since)
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
