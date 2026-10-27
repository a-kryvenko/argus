"""Normalize stored observations independently of ingestion and forecasting."""
import logging

from clio.db.session import get_db_session
from clio.ingestion.normalize import refresh_normalized_observations

logger = logging.getLogger(__name__)


async def run(args) -> None:
    async with get_db_session() as session:
        observations = await refresh_normalized_observations(session, now=args.now)
        logger.info("Updated %d normalized observations", len(observations.points))
