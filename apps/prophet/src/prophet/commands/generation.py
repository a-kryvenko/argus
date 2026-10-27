"""One-shot forecast generation with supervised calculation processes."""
import logging


async def run(args):
    from prophet.services.generation.cycle import generate
    from prophet.db.locks import generation_lock
    from prophet.execution import supervise, ShutdownRequested
    from prophet.config import load_config

    try:
        with supervise(load_config().shutdown_grace_seconds):
            async with generation_lock() as writer:
                await generate(args.product, writer=writer)
    except ShutdownRequested:
        logging.info('Prophet generation stopped')
        raise SystemExit(130)
