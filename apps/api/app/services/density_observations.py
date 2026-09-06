"""Build JB2008-only derived inputs; never write to shared observations."""
import asyncio
import json
import logging

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from forecast.inference.jb2008_drivers import (
    BACKGROUND_METHOD, DTC_METHOD, SOURCE_METRICS, SOLAR_LAGS_DAYS,
    DriverDataUnavailable, prepare_snapshot,
)

from app.db.models import Measurement
from app.services.forecast_products import ArtifactNotReadyError

logger = logging.getLogger(__name__)

DRIVER_METRICS = SOURCE_METRICS


def observed_driver_frame(measurements: pd.DataFrame, issue_time: datetime) -> pd.DataFrame:
    try:
        snapshot = prepare_snapshot(measurements, issue_time)
    except DriverDataUnavailable as exc:
        raise ArtifactNotReadyError(str(exc)) from exc
    issue = pd.Timestamp(issue_time)
    return pd.DataFrame([
        {**snapshot.values, "observed_at": snapshot.observed_at,
         "history_start": snapshot.history_start,
         "dtc_observed_at": snapshot.dtc_observed_at,
         "background_method": (BACKGROUND_METHOD + "_linear_gapfill"
                               if snapshot.background_interpolated_days else BACKGROUND_METHOD),
         "background_interpolated_days": json.dumps(snapshot.background_interpolated_days, sort_keys=True),
         "dtc_method": DTC_METHOD,
         "valid_time": issue + pd.Timedelta(hours=lead)}
        for lead in range(49)
    ])


async def load_density_drivers(session: AsyncSession, issue_time: datetime) -> pd.DataFrame:
    logger.info("JB2008: querying 88 days of internal observations")
    result = await asyncio.wait_for(session.execute(
        select(Measurement.metric, Measurement.value, Measurement.observed_at)
        .where(Measurement.metric.in_(SOURCE_METRICS),
               Measurement.observed_at >= issue_time - timedelta(days=88),
               Measurement.observed_at <= issue_time)
        .order_by(Measurement.observed_at)
    ), timeout=60)
    records = pd.DataFrame(result.all(), columns=["metric", "value", "observed_at"])
    logger.info("JB2008: loaded %d observation rows; preparing drivers", len(records))
    try:
        return observed_driver_frame(records, issue_time)
    except ArtifactNotReadyError as exc:
        if not any(name in str(exc) for name in SOLAR_LAGS_DAYS):
            raise
        logger.info("JB2008: supplementing solar observations from private history cache")
        from app.services.density_history import load_density_history, merge_history
        try:
            history = await asyncio.to_thread(load_density_history, issue_time)
        except (OSError, KeyError, ValueError) as history_error:
            raise ArtifactNotReadyError(f"JB2008 history unavailable: {history_error}") from history_error
        return observed_driver_frame(merge_history(history, records), issue_time)
