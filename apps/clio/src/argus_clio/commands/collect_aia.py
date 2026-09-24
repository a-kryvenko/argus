"""Collect every hourly AIA193 slot, retaining original FITS and receipt time."""
import argparse
import asyncio

from argus_clio.services.aia.collection import collect_aia
from argus_clio.db.session import dispose_engine


async def collect(history_days: int) -> None:
    try:
        await collect_aia(history_days=history_days)
    finally:
        await dispose_engine()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-days', type=int, default=40)
    args = parser.parse_args(argv)
    asyncio.run(collect(args.history_days))
