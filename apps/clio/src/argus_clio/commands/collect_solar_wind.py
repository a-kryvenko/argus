"""Collect minute solar wind samples once, or continuously with --watch."""
import argparse
import asyncio
import logging
from time import monotonic

from argus_clio.commands._runner import run_command
from argus_clio.commands._shutdown import stop_on_signal, wait_for_next_poll
from argus_clio.db.session import dispose_engine
from argus_clio.services.collection.heartbeat import CollectorHeartbeat
from argus_clio.services.solar_wind.observations import refresh_solar_wind
from argus_clio.services.collection.specs import WIND_POLL_SECONDS

logger = logging.getLogger(__name__)


async def collect(watch: bool) -> None:
    heartbeat = CollectorHeartbeat('solar-wind') if watch else None
    try:
        with stop_on_signal(watch) as stopped:
            while not stopped.is_set():
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
                await wait_for_next_poll(stopped, max(1, WIND_POLL_SECONDS - (monotonic() - started)))
    finally:
        if heartbeat:
            heartbeat.stop()
        await dispose_engine()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help=f"Poll each source every {WIND_POLL_SECONDS} seconds")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect(args.watch))


if __name__ == "__main__":
    run_command(main)
