"""Collect minute solar wind samples once, or continuously with --watch."""
import argparse
import asyncio
import logging
from time import monotonic

from argus_clio.commands._runner import run_command
from argus_clio.db.session import dispose_engine
from argus_clio.services.collector_heartbeat import CollectorHeartbeat
from argus_clio.services.solar_wind import refresh_solar_wind

logger = logging.getLogger(__name__)


async def collect(watch: bool) -> None:
    heartbeat = CollectorHeartbeat('solar-wind') if watch else None
    try:
        while True:
            started = monotonic()
            if heartbeat:
                for source_id in heartbeat.sources:
                    heartbeat.started(source_id)
            try:
                await refresh_solar_wind()
            except Exception:
                if not watch:
                    raise
                logger.exception("Solar wind collection incomplete; retrying next cycle")
            finally:
                if heartbeat:
                    for source_id in heartbeat.sources:
                        heartbeat.finished(source_id)
            if not watch:
                return
            await asyncio.sleep(max(1, 60 - (monotonic() - started)))
    finally:
        if heartbeat:
            heartbeat.stop()
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Poll each source every 60 seconds")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect(args.watch))


if __name__ == "__main__":
    run_command(main)
