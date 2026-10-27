"""Bounded, resumable export; legacy tables remain unchanged."""
import logging

from clio.observations.transition import backfill, verify


async def run(args):
    result = await verify() if args.check else await backfill(batch_size=args.batch_size)
    logging.getLogger(__name__).info('File archive: %s', result)
    return int(any(result.values()) if args.check else bool(result['failed']))
