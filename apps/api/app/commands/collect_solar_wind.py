"""Collect minute solar wind samples once, or continuously with --watch."""
import argparse
import asyncio
import logging
from time import monotonic

from app.commands._runner import run_command
from app.db.session import dispose_engine
from app.services.solar_wind import refresh_solar_wind

logger = logging.getLogger(__name__)


async def collect(watch: bool) -> None:
    try:
        while True:
            started = monotonic()
            try:
                await refresh_solar_wind()
            except Exception:
                if not watch:
                    raise
                logger.exception("Solar wind collection incomplete; retrying next cycle")
            if not watch:
                return
            await asyncio.sleep(max(1, 60 - (monotonic() - started)))
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Poll each source every 60 seconds")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect(args.watch))


if __name__ == "__main__":
    run_command(main)
