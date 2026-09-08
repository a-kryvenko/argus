"""Process durable five-minute/hourly aggregation backlog without fetching sources."""
import argparse
import asyncio
import logging
from app.commands._runner import run_command
from app.db.session import dispose_engine
from app.services.solar_wind_aggregation import aggregate_pending


async def collect(limit):
    try:
        count = await aggregate_pending(limit)
        logging.info('Aggregated %d source hours', count)
    finally:
        await dispose_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=240, help='Maximum source hours per run (default 240)')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect(args.limit))


if __name__ == '__main__':
    run_command(main)
