import asyncio
import json
from time import monotonic


async def _run(args):
    from common.config import get_config
    from prophet.services.verification import verify, verification_report
    get_config()
    if args.command == 'verify':
        from prophet.config import load_config
        from prophet.execution import supervise, ShutdownRequested
        from prophet.db.locks import verification_lock
        try:
            with supervise(load_config().shutdown_grace_seconds):
                async with verification_lock() as writer:
                    timeout = load_config().verification_timeout_seconds
                    async with asyncio.timeout(timeout):
                        result = await verify(args.product, writer=writer, days=args.days, deadline=monotonic()+timeout)
        except ShutdownRequested:
            raise SystemExit(130)
    else:
        result = await verification_report(args.product, days=args.days)
    print(json.dumps(result, default=str, indent=2, allow_nan=False))


def run(args):
    from prophet.commands.runner import run_async
    return run_async(_run(args))
