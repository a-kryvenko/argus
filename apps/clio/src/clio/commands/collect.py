"""Run collect and print its report."""
import json

from clio.config import load_observation_config


async def run(args):
    from clio.ingestion.collect import collect
    result = await collect(load_observation_config(), args.metrics, now=args.now)
    print(json.dumps(result, indent=2))
    return result
