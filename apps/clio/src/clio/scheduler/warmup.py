import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent

from clio.scheduler.jobs import run_backfill, run_collect

logger = logging.getLogger(__name__)

def read_state(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def write_state(path: Path, state):
    state["last_checked_at"] = datetime.now(UTC).isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def needs_warmup(previous):
    if previous is None or previous.get("warmup_pending"):
        return True

    last_checked_at = datetime.fromisoformat(previous["last_checked_at"])
    return datetime.now(UTC) - last_checked_at > timedelta(minutes=1)


async def heartbeat(path: Path, state):
    while True:
        await asyncio.sleep(10)
        write_state(path, state)


async def _run_job(job_id, func, report_job, *args):
    started_at = datetime.now(UTC)
    logger.info("Running warmup job %s", job_id)
    try:
        result = await func(*args)
    except Exception as exc:
        logger.exception("Warmup job %s failed", job_id)
        event = JobExecutionEvent(
            EVENT_JOB_ERROR, job_id, "default", started_at, exception=exc,
        )
    else:
        event = JobExecutionEvent(
            EVENT_JOB_EXECUTED, job_id, "default", started_at, retval=result,
        )
    report_job(event)


async def run(collect_groups, report_job):
    logger.info("Warmup started")
    await _run_job("warmup:backfill", run_backfill, report_job)
    for metrics in collect_groups:
        await _run_job(
            "warmup:collect:" + "|".join(metrics), run_collect, report_job, metrics,
        )
    logger.info("Warmup finished")
