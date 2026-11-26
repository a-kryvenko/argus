import argparse
import asyncio
import logging
import signal
from datetime import UTC, datetime
from pathlib import Path

import sentry_sdk
from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_SCHEDULER_SHUTDOWN,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from common.runtime import setup_sentry

from clio.config import load_observation_config
from clio.scheduler import warmup
from clio.scheduler.jobs import restore_donki, run_backfill, run_cleanup, run_collect, run_donki, run_normalize
from clio.scheduler.schedule import collecting_schedules

logger = logging.getLogger(__name__)


def configure_jobs(scheduler, config, *, warmup_required):
    def schedule(func, trigger, job_id, *, args=(), priority=False):
        options = {"next_run_time": datetime.now(UTC)} if priority and warmup_required else {}
        scheduler.add_job(func, trigger, id=job_id, args=args, **options)

    for metrics, minutes in collecting_schedules(config):
        schedule(run_collect, IntervalTrigger(minutes=minutes, timezone=UTC),
                 "collect:" + "|".join(metrics), args=(metrics,), priority=True)
    if config.solar_wind_events.enabled:
        schedule(restore_donki, DateTrigger(timezone=UTC), "restore-donki")
        schedule(run_donki, IntervalTrigger(minutes=config.solar_wind_events.poll_minutes, timezone=UTC),
                 "donki", args=(config.solar_wind_events,), priority=True)
    schedule(run_backfill, CronTrigger(hour="*/6", minute=0, timezone=UTC), "backfill", priority=True)
    schedule(run_normalize, CronTrigger(minute=0, timezone=UTC), "normalize")
    schedule(run_cleanup, CronTrigger(minute=15, timezone=UTC), "cleanup")


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

        if result.get("status") == "busy":
            logger.info("Job %s reported an existing collection in progress", event.job_id)

        incomplete = (
            result.get("status") in {"partial", "failed"}
            or bool(result.get("failed_metrics"))
            or bool(result.get("failed"))
        )

        errors = any(attempt.get("error") for attempt in result.get("source_attempts", ()))
        if incomplete and (event.job_id != "backfill" or errors or result.get("failed_metrics")):
            details = {
                "status": result.get("status"),
                "failed_metrics": result.get("failed_metrics"),
                "failed": result.get("failed"),
                "failed_endpoints": result.get("failed_endpoints"),
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
    required = warmup.needs_warmup(warmup.read_state(heartbeat_path))
    state = {"warmup_pending": required, "scheduler_started": False, "jobs": {}}
    warmup.write_state(heartbeat_path, state)
    scheduler = AsyncIOScheduler(
        timezone=UTC,
        job_defaults={
            "misfire_grace_time": None,
            "coalesce": True,
            "max_instances": 1,
        },
    )
    def report(event):
        status = "failed" if event.code == EVENT_JOB_ERROR else "complete"
        if isinstance(event.retval, dict):
            status = event.retval.get("status", status)
        if event.job_id == "backfill" and status in {"complete", "partial"}:
            state["warmup_pending"] = False
        state["jobs"][event.job_id] = {"status": status, "finished_at": datetime.now(UTC).isoformat()}
        on_job_finished(event)

    scheduler.add_listener(report, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)
    configure_jobs(scheduler, load_observation_config(), warmup_required=required)

    scheduler_stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, main_task.cancel)

    scheduler.add_listener(
        lambda event: scheduler_stopped.set(),
        EVENT_SCHEDULER_SHUTDOWN,
    )

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
        scheduler.start()
        state["scheduler_started"] = True
        warmup.write_state(heartbeat_path, state)
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Worker stopped")
    finally:
        state["scheduler_started"] = False
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
