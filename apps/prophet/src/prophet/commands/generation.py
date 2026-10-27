"""One-shot forecast generation with supervised calculation processes."""
import logging


def run(args):
    from prophet.services.generation.cycle import generate
    from common.runtime import run_command
    from prophet.commands.runner import run_async
    from prophet.db.locks import generation_lock
    from prophet.execution import supervise, ShutdownRequested
    from prophet.config import load_config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    async def once():
        try:
            with supervise(load_config().shutdown_grace_seconds):
                async with generation_lock() as writer:
                    await generate(args.product, writer=writer)
        except ShutdownRequested:
            logging.info('Prophet generation stopped')
            raise SystemExit(130)
    return run_command(lambda: run_async(once()))
