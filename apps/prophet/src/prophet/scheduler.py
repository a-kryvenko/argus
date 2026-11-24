"""Schedule Prophet's one-shot commands in a single worker."""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger
from common.scheduler import WorkerScheduler

from prophet.services.generation.products import PRODUCTS


logger = logging.getLogger(__name__)


def register_jobs(worker):
    writer = asyncio.Lock()

    async def run(job_id, command):
        async with writer:
            if not worker.stop.is_set():
                await run_command(job_id, command, worker.stop)

    def schedule(job_id, command, *, minute, hour='*'):
        worker.add_job(run, CronTrigger(minute=minute, hour=hour, timezone=worker.scheduler.timezone),
                       id=job_id, args=(job_id, command))

    for product in PRODUCTS:
        schedule(f'generation:{product}', ('generation', product),
                 minute='*/5' if product == 'solar-wind-events' else 10)
    schedule('summary', ('summary',), minute='*/5')
    schedule('verification', ('verification', 'all'), minute=30, hour='*/6')
    schedule('cleanup', ('cleanup', '--apply'), minute=45)


async def run_command(job_id, command, stop):
    logger.info('Running %s', job_id)
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'prophet.cli', *command,
        env={**os.environ, 'ARGUS_SCHEDULED_JOB': job_id, 'PYTHONUNBUFFERED': '1'},
    )
    completed = asyncio.create_task(process.wait())
    stopping = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait((completed, stopping), return_when=asyncio.FIRST_COMPLETED)
        if stopping in done and not completed.done() and command[0] in ('generation', 'verification'):
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        status = await completed
        if status and not (stop.is_set() and status in (-15, 130)):
            raise RuntimeError(f'Prophet job {job_id} exited with status {status}')
    finally:
        stopping.cancel()
        await asyncio.gather(stopping, return_exceptions=True)


async def main(heartbeat_file):
    worker = WorkerScheduler()
    register_jobs(worker)
    await worker.serve(heartbeat_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heartbeat-file', required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    asyncio.run(main(args.heartbeat_file))
