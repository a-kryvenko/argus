import asyncio
from datetime import UTC, datetime, timedelta

from common.schemas.observation import Observation

from app.db.session import dispose_engine, get_session_factory
from app.services.sensor_observations import HISTORY_DAYS, load_normalized_observations


async def _load() -> Observation:
    try:
        async with get_session_factory()() as session:
            observations = await load_normalized_observations(
                session, since=datetime.now(UTC) - timedelta(days=HISTORY_DAYS),
            )
            if not observations.points:
                raise RuntimeError(
                    "No stored observations available in the last 30 days. "
                    "Run app.commands.refresh_observations first."
                )
            return observations
    finally:
        await dispose_engine()


def load_sensor_observations() -> Observation:
    return asyncio.run(_load())
