"""Prepare the rolling verification documents served by Prophet API."""
import asyncio
import logging
from time import monotonic


async def run(args):
    from common.config import get_config
    from prophet.config import load_config
    from prophet.execution import ShutdownRequested, supervise
    from prophet.services.heavy_task import wait_for_heavy_task
    from prophet.services.verification import publish_monthly_accuracy

    config = load_config()
    timeout = config.verification_timeout_seconds
    try:
        with supervise(config.shutdown_grace_seconds):
            deadline = monotonic() + timeout
            async with asyncio.timeout(timeout), wait_for_heavy_task(get_config().data_root):
                count = await publish_monthly_accuracy(deadline=deadline)
    except ShutdownRequested:
        raise SystemExit(130)
    logging.info('Prepared rolling verification for %d forecast products', count)
