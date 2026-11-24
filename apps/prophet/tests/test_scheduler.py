import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from common.scheduler import WorkerScheduler
from prophet import scheduler
from prophet.services.generation.products import PRODUCTS


def test_prophet_scheduler_preserves_products_and_cadences(monkeypatch):
    async def check():
        worker = WorkerScheduler()
        run = AsyncMock()
        monkeypatch.setattr(scheduler, 'run_command', run)
        scheduler.register_jobs(worker)
        jobs = {job.id: job for job in worker.scheduler.get_jobs()}
        assert set(jobs) == {*(f'generation:{product}' for product in PRODUCTS), 'summary', 'verification', 'cleanup'}
        now = datetime(2026, 11, 24, 1, 1, tzinfo=UTC)
        for job_id, job in jobs.items():
            if job_id.startswith('generation:'):
                product = job_id.split(':', 1)[1]
                command = ('generation', product)
                expected = (1, 5 if product == 'solar-wind-events' else 10)
            else:
                command, expected = {
                    'summary': (('summary',), (1, 5)),
                    'verification': (('verification', 'all'), (6, 30)),
                    'cleanup': (('cleanup', '--apply'), (1, 45)),
                }[job_id]
            fire = job.trigger.get_next_fire_time(None, now)
            assert (fire.hour, fire.minute) == expected
            await job.func()
            run.assert_awaited_with(job_id, command, worker.stop)

    asyncio.run(check())


def test_scheduler_skips_jobs_waiting_for_the_writer_during_shutdown(monkeypatch):
    async def check():
        worker = WorkerScheduler()
        active = asyncio.Event()

        async def run(job_id, command, stop):
            active.set()
            await stop.wait()

        command = AsyncMock(side_effect=run)
        monkeypatch.setattr(scheduler, 'run_command', command)
        scheduler.register_jobs(worker)
        first = asyncio.create_task(worker.scheduler.get_job('generation:dst').func())
        await active.wait()
        queued = asyncio.create_task(worker.scheduler.get_job('summary').func())
        await asyncio.sleep(0)
        worker.stop.set()
        await asyncio.gather(first, queued)
        command.assert_awaited_once_with('generation:dst', ('generation', 'dst'), worker.stop)

    asyncio.run(check())
