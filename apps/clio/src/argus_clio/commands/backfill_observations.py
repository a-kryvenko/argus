"""Backfill missing measurements and rebuild hourly observations in a UTC range."""
import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json

from argus_clio.commands.audit_solar_wind import utc_hour
from argus_clio.db.session import dispose_engine, get_session_factory
from argus_clio.services.backfill import backfill


def boundary(value):
    if len(value) == 10:
        value += 'T00:00:00Z'
    return utc_hour(value)


async def run(start, end):
    try:
        async with get_session_factory()() as session:
            result = await backfill(session, start, end)
        print(json.dumps(result, indent=2))
    finally:
        await dispose_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='start', type=boundary, required=True, help='Inclusive UTC date or whole hour')
    parser.add_argument('--to', dest='end', type=boundary, required=True, help='Exclusive UTC date or whole hour')
    args = parser.parse_args()
    if not timedelta(0) < args.end - args.start <= timedelta(days=31) or args.end > datetime.now(UTC):
        parser.error('Choose a past range of at most 31 days, with --from before --to')
    asyncio.run(run(args.start, args.end))
