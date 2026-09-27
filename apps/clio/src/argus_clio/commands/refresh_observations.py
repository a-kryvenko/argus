"""Ingest observations independently of forecast generation."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta

from argus_clio.services.observations.density_history import load_density_history, merge_history

from argus_clio.db.session import get_session_factory
from argus_clio.services.observations.normalized import refresh_normalized_observations
from argus_clio.services.observations.store import load_measurements, upsert_measurements

logger = logging.getLogger(__name__)


async def run(args) -> None:
    now = datetime.now(UTC)
    async with get_session_factory()() as session:
        observations = await refresh_normalized_observations(session, now=now)
        logger.info("Updated %d normalized observations", len(observations.points))
        # JB2008 needs a longer solar history than other forecast products.
        # Missing optional calibration/history must not roll back live data.
        try:
            history = await asyncio.to_thread(load_density_history, now)
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Density observation history unavailable: %s", exc)
        else:
            existing = await load_measurements(session, since=now - timedelta(days=88))
            await upsert_measurements(session, merge_history(history, existing), track_receipt=False)
            await session.commit()
