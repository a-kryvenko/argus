"""Run backfill and print its report."""
import json

from clio.config import load_observation_config


async def run(args):
    from clio.ingestion.backfill import backfill
    result = await backfill(load_observation_config(), now=args.now)
    print(json.dumps(result, indent=2))
    return result
