import asyncio

from common.schemas.observation import Observation

from app.db.session import dispose_engine, get_session_factory
from app.services.sensor_observations import refresh_normalized_observations


async def _refresh() -> Observation:
    try:
        async with get_session_factory()() as session:
            return await refresh_normalized_observations(session)
    finally:
        await dispose_engine()


def refresh_sensor_observations() -> Observation:
    return asyncio.run(_refresh())
