"""Build hourly observations from retained live measurements."""
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from clio.config import load_observation_config
from clio.ingestion.products import PRODUCTS
from clio.observations.measurements import load_measurements
from clio.observations.normalized import load_normalized_observations, upsert_normalized_observations
from clio.observations.schema import REQUIRED_METRICS, SOLAR_INDEX_METRICS, OBSERVATION_METRICS
from common.schemas.observation import Observation


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
    session: AsyncSession,
    now: datetime | None = None,
) -> Observation:
    """Write hourly rows from retained live measurements."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    config = load_observation_config()
    since = now - timedelta(days=config.retention.live_days)
    products = [name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes]
    measurements = await load_measurements(session, since, until=now + timedelta(microseconds=1),
                                           source_products=products)
    normalized = normalize_measurements(measurements)
    if normalized.empty:
        raise RuntimeError("No complete normalized observations could be built; run collect first")

    await upsert_normalized_observations(session, normalized)
    await session.commit()
    return await load_normalized_observations(session, since=since, until=now)
