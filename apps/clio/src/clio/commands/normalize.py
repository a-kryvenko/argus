"""Normalize stored observations independently of ingestion and forecasting."""
import logging

from clio.ingestion.normalize import refresh_normalized_observations
from clio.config import load_observation_config

logger = logging.getLogger(__name__)


async def run(args) -> None:
    observations = await refresh_normalized_observations(
        load_observation_config(),
        now=args.now
    )
    logger.info("Updated %d normalized observations", len(observations.points))
