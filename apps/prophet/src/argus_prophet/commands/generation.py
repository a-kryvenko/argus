"""Manual and scheduled generation share one cycle and process supervisor."""
import logging


def run(args):
    from argus_prophet.services.generation.cycle import generate
    from common.runtime import run_command
    from argus_prophet.worker import work
    from argus_prophet.scheduling.jobs import generation_lock
    from argus_prophet.scheduling.execution import supervise, ShutdownRequested
    from argus_prophet.config import load_config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'worker':
        return run_command(work)

    def once():
        try:
            with supervise(load_config().shutdown_grace_seconds), generation_lock() as writer:
                generate(args.product, writer=writer)
        except ShutdownRequested:
            logging.info('Prophet generation stopped')
            raise SystemExit(130)
    return run_command(once)
