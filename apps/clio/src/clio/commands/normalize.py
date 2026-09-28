"""Normalize stored observations independently of ingestion and forecasting."""
import logging
from datetime import UTC, datetime

from clio.db.session import get_session_factory
from clio.observations.normalized import refresh_normalized_observations

logger = logging.getLogger(__name__)


async def run(args) -> None:
    now = datetime.now(UTC)
    async with get_session_factory()() as session:
        observations = await refresh_normalized_observations(session, now=now)
        logger.info("Updated %d normalized observations", len(observations.points))
