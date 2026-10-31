"""Remove expired live measurements, hourly observations and SDO files."""

import json

from clio.config import load_observation_config
from clio.ingestion.cleanup import cleanup as _cleanup


async def run(args):
    result = await _cleanup(load_observation_config(), args.now)
    
    print(json.dumps(result, indent=2))
    return result
