"""Build JB2008-only derived inputs; never write to shared observations."""
import asyncio
import logging

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from forecast_core.api import (
    SOURCE_METRICS, SOLAR_LAGS_DAYS, DriverDataUnavailable, prepare_density_drivers,
)

from app.db.models import Measurement
from app.services.forecast_products import ArtifactNotReadyError

logger = logging.getLogger(__name__)

DRIVER_METRICS = SOURCE_METRICS


def observed_driver_frame(measurements: pd.DataFrame, issue_time: datetime) -> pd.DataFrame:
    try:
        return prepare_density_drivers(measurements, issue_time)
    except DriverDataUnavailable as exc:
        raise ArtifactNotReadyError(str(exc)) from exc


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
    return observed_driver_frame(records, issue_time)
