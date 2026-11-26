import asyncio
from datetime import UTC, datetime

from clio.config import load_observation_config


async def run_collect(metrics: list):
    from clio.ingestion.collect import collect

    result = await collect(load_observation_config(), metrics, now=datetime.now(UTC))
    return result

async def run_donki(policy):
    from clio.ingestion.extras.donki import collect

    return await collect(policy, now=datetime.now(UTC))

async def restore_donki():
    from clio.ingestion.extras.donki import restore_legacy_polls

    while True:
        result = await asyncio.to_thread(restore_legacy_polls)
        if result['status'] != 'busy':
            return result
        await asyncio.sleep(5)

async def run_backfill():
    from clio.ingestion.backfill import backfill

    result = await backfill(load_observation_config(), now=datetime.now(UTC))
    return result

async def run_normalize():
    from clio.ingestion.normalize import refresh_normalized_observations

    result = await refresh_normalized_observations(
        load_observation_config(),
        now=datetime.now(UTC)
    )
    return result

async def run_cleanup():
    from clio.ingestion.cleanup import cleanup

    result = await cleanup(load_observation_config(), now=datetime.now(UTC))
    return result
