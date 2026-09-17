"""Ingest observations independently of forecast generation."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta

from argus_clio.services.calibration import load_density_history, merge_history

from argus_clio.commands._runner import run_command
from argus_clio.db.session import dispose_engine, get_session_factory
from argus_clio.services.sensor_observations import (
    _load_measurements,
    _upsert_measurements,
    refresh_normalized_observations,
)

logger = logging.getLogger(__name__)


async def _refresh() -> None:
    now = datetime.now(UTC)
    try:
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
                existing = await _load_measurements(session, since=now - timedelta(days=88))
                await _upsert_measurements(session, merge_history(history, existing), track_receipt=False)
                await session.commit()
    finally:
        await dispose_engine()


def main() -> None:
    asyncio.run(_refresh())


if __name__ == "__main__":
    run_command(main)
