import argparse
import asyncio
import logging
import signal
from pathlib import Path

import sentry_sdk
from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_SCHEDULER_SHUTDOWN,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from common.runtime import setup_sentry

from clio.scheduler import warmup
from clio.scheduler.jobs import run_backfill, run_cleanup, run_collect, run_normalize
from clio.scheduler.schedule import collecting_schedules
from clio.config import load_observation_config

logger = logging.getLogger(__name__)


def _init_job(scheduler, func, job_id, args=(), **schedule):
    options = {
        "func": func,
        "args": args,
        "trigger": CronTrigger(timezone=scheduler.timezone, **schedule),
        "name": job_id,
    }
    if scheduler.get_job(job_id) is None:
        scheduler.add_job(id=job_id, **options)
    else:
        scheduler.modify_job(job_id, **options)


def _init_collect_job(scheduler, metrics, minutes):
    job_id = 'collect:' + '|'.join(metrics)
    options = {
        "func": run_collect,
        "args": (metrics,),
        "trigger": IntervalTrigger(minutes=minutes, timezone=scheduler.timezone),
        "name": job_id
    }
    if scheduler.get_job(job_id) is None:
        scheduler.add_job(id=job_id, **options)
    else:
        scheduler.modify_job(job_id, **options)


def on_job_finished(event):
    with sentry_sdk.isolation_scope() as scope:
        scope.set_tag("service", "clio")
        scope.set_tag("job_id", event.job_id)

        if event.code == EVENT_JOB_ERROR:
            if isinstance(event.exception, asyncio.CancelledError):
                return

            sentry_sdk.capture_exception(event.exception)
            return

        result = event.retval
        if not isinstance(result, dict):
            return

        incomplete = (
            result.get("status") in {"partial", "failed"}
            or bool(result.get("failed_metrics"))
            or bool(result.get("failed"))
        )

        if incomplete:
            details = {
                "status": result.get("status"),
                "failed_metrics": result.get("failed_metrics"),
                "failed": result.get("failed")
            }
            logger.warning(
                "Job %s completed with failures: %s",
                event.job_id,
                details
            )
            scope.set_context("job_result", details)
            sentry_sdk.capture_message(
                "Scheduler job completed with failures",
                level="warning"
            )

def entrypoint(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat-file", type=Path, required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        setup_sentry()
        asyncio.run(main(args.heartbeat_file))
    except Exception as exc:
        logger.exception("Scheduler worker failed")
        sentry_sdk.capture_exception(exc)
        raise SystemExit(1) from None
    finally:
        sentry_sdk.flush(timeout=2)

async def main(heartbeat_path: Path):
    scheduler = AsyncIOScheduler(
        job_defaults={
            "misfire_grace_time": None,
            "coalesce": True,
            "max_instances": 1,
        },
    )
    scheduler.add_listener(
        on_job_finished,
        EVENT_JOB_ERROR | EVENT_JOB_EXECUTED,
    )

    """
    The Core of worker.
    Scheduled jobs
    """
    schedules = collecting_schedules(load_observation_config())
    for metrics, minutes in schedules:
        _init_collect_job(scheduler, metrics, minutes)

    _init_job(scheduler, run_backfill, "backfill", hour="*/6", minute=0)
    _init_job(scheduler, run_normalize, "normalize", minute=0)
    _init_job(scheduler, run_cleanup, "cleanup", minute=15)
    
    scheduler_stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, main_task.cancel)

    scheduler.add_listener(
        lambda event: scheduler_stopped.set(),
        EVENT_SCHEDULER_SHUTDOWN,
    )

    """
    Hearthbeat.
    To be able warm up whole application after long downtime.
    Run regular backfill + all metrics collecting
    """
    required = warmup.needs_warmup(warmup.read_state(heartbeat_path))
    state = {"warmup_pending": required}
    warmup.write_state(heartbeat_path, state)

    heartbeat_task = asyncio.create_task(warmup.heartbeat(heartbeat_path, state))

    def on_heartbeat_finished(task):
        if (
            not task.cancelled()
            and task.exception() is not None
            and not main_task.cancelling()
        ):
            main_task.cancel()

    heartbeat_task.add_done_callback(on_heartbeat_finished)

    try:
        if required:
            await warmup.run(
                [metrics for metrics, _ in schedules],
                on_job_finished
            )
            state["warmup_pending"] = False
            warmup.write_state(heartbeat_path, state)

        scheduler.start()
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Worker stopped")
    finally:
        if scheduler.running:
            logger.info("Stopping scheduler...")
            scheduler.shutdown(wait=False)
            await scheduler_stopped.wait()

        heartbeat_task.cancel()
        heartbeat_result, = await asyncio.gather(heartbeat_task, return_exceptions=True)

        if isinstance(heartbeat_result, Exception):
            raise heartbeat_result
        else:
            warmup.write_state(heartbeat_path, state)

if __name__ == '__main__':
    entrypoint()
