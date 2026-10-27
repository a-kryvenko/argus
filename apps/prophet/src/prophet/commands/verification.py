"""Verify published forecasts against later observations."""
import asyncio
import json
from time import monotonic


async def run(args):
    from prophet.services.verification import verify
    from prophet.config import load_config
    from prophet.execution import supervise, ShutdownRequested
    from prophet.db.locks import verification_lock

    config = load_config()
    try:
        with supervise(config.shutdown_grace_seconds):
            async with verification_lock() as writer:
                timeout = config.verification_timeout_seconds
                async with asyncio.timeout(timeout):
                    result = await verify(args.product, writer=writer, deadline=monotonic()+timeout)
    except ShutdownRequested:
        raise SystemExit(130)
    print(json.dumps(result, default=str, indent=2, allow_nan=False))
