"""Collect native Kp/Dst once, or poll independently with --watch."""
import argparse
import asyncio
import logging
from time import monotonic

from app.commands._runner import run_command
from app.db.session import dispose_engine
from app.services.collector_heartbeat import CollectorHeartbeat
from app.services.geomagnetic import ingest_source, refresh_geomagnetic
from clio.dataloaders.geomagnetic_loader import POLL_SECONDS

logger = logging.getLogger(__name__)


async def watch_source(metric: str, heartbeat: CollectorHeartbeat | None = None) -> None:
    while True:
        started = monotonic()
        if heartbeat:
            heartbeat.started(metric)
        try:
            await ingest_source(metric)
        except Exception:
            logger.exception('%s collection failed; retrying next cycle', metric)
        finally:
            if heartbeat:
                heartbeat.finished(metric)
        await asyncio.sleep(max(1, POLL_SECONDS[metric] - (monotonic() - started)))


async def collect(watch: bool) -> None:
    heartbeat = None
    try:
        if watch:
            heartbeat = CollectorHeartbeat('geomagnetic')
            await asyncio.gather(*(watch_source(metric, heartbeat) for metric in POLL_SECONDS))
        else:
            await refresh_geomagnetic()
    finally:
        if heartbeat:
            heartbeat.stop()
        await dispose_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect(args.watch))


if __name__ == '__main__':
    run_command(main)
