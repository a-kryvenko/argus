import asyncio
import json
import signal
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from apscheduler.events import EVENT_JOB_EXECUTED
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from clio.scheduler import warmup, worker


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "worker-state.json"


@pytest.fixture(autouse=True)
def donki_job(monkeypatch):
    job = AsyncMock(return_value={"status": "complete"})
    monkeypatch.setattr(worker, "run_donki", job)
    monkeypatch.setattr(worker, "restore_donki", AsyncMock(return_value={"status": "complete"}))
    return job


def test_worker_requires_heartbeat_file(monkeypatch):
    setup = Mock()
    monkeypatch.setattr(worker, "setup_sentry", setup)
    with pytest.raises(SystemExit) as caught:
        worker.entrypoint([])
    assert caught.value.code == 2
    setup.assert_not_called()


def test_worker_passes_heartbeat_file_to_main(monkeypatch, tmp_path):
    path = tmp_path / "custom state.json"
    main = AsyncMock()
    monkeypatch.setattr(worker, "main", main)
    monkeypatch.setattr(worker, "setup_sentry", Mock())
    monkeypatch.setattr(worker.sentry_sdk, "flush", Mock())
    worker.entrypoint(["--heartbeat-file", str(path)])
    main.assert_awaited_once_with(path)


@pytest.mark.parametrize("seconds,pending,expected", [
    (10, False, False),
    (60, False, False),
    (61, False, True),
    (10, True, True),
])
def test_warmup_threshold(monkeypatch, seconds, pending, expected):
    now = datetime(2026, 10, 31, tzinfo=UTC)
    monkeypatch.setattr(warmup, "datetime", SimpleNamespace(
        now=lambda tz: now, fromisoformat=datetime.fromisoformat,
    ))
    previous = {
        "last_checked_at": (now - timedelta(seconds=seconds)).isoformat(),
        "warmup_pending": pending,
    }
    assert warmup.needs_warmup(previous) is expected
    assert warmup.needs_warmup(None)


def test_cold_start_collects_while_archive_is_blocked(monkeypatch, state_path, donki_job):
    from clio.scheduler.healthcheck import ready

    async def check():
        collecting = asyncio.Event()
        backfilling = asyncio.Event()
        completed = asyncio.Event()
        cancelled = asyncio.Event()
        signals = {}

        async def backfill():
            assert ready(state_path)
            backfilling.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def collect(metrics):
            assert metrics == ["v"]
            collecting.set()
            return {"status": "complete"}

        def report(event):
            if event.job_id == 'collect:v':
                assert event.code == EVENT_JOB_EXECUTED
                completed.set()

        monkeypatch.setattr(worker, "run_backfill", backfill)
        monkeypatch.setattr(worker, "run_collect", collect)
        monkeypatch.setattr(worker, "collecting_schedules", lambda config: [(["v"], 1)])
        monkeypatch.setattr(worker, "on_job_finished", report)
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: signals.update({sig: callback}))

        task = asyncio.create_task(worker.main(state_path))
        await asyncio.wait_for(asyncio.gather(collecting.wait(), backfilling.wait(), completed.wait()), timeout=2)
        assert ready(state_path) and warmup.read_state(state_path)["warmup_pending"]
        assert donki_job.await_count == 1
        signals[signal.SIGTERM]()
        await asyncio.wait_for(task, timeout=2)
        assert cancelled.is_set()
        assert not ready(state_path)
        assert warmup.read_state(state_path)["jobs"]["collect:v"]["status"] == "complete"

    asyncio.run(check())


@pytest.mark.parametrize("warmup_required", [True, False])
def test_scheduler_sets_initial_run_time_and_recurrence(warmup_required):
    async def check():
        scheduler = AsyncIOScheduler(timezone=UTC)
        config = worker.load_observation_config()
        before = datetime.now(UTC)
        worker.configure_jobs(scheduler, config, warmup_required=warmup_required)
        scheduler.start(paused=True)
        try:
            assert len(scheduler.get_jobs()) == len(worker.collecting_schedules(config)) + 5
            for job in (scheduler.get_job("backfill"), scheduler.get_job("donki")):
                if warmup_required:
                    assert before <= job.next_run_time <= datetime.now(UTC)
                else:
                    assert job.next_run_time > datetime.now(UTC)
            assert scheduler.get_job("donki").trigger.interval.total_seconds() == config.solar_wind_events.poll_minutes * 60
        finally:
            scheduler.shutdown()
            await asyncio.sleep(0)

    asyncio.run(check())


def test_healthcheck_rejects_recent_heartbeat_before_scheduler_started(state_path):
    from clio.scheduler.healthcheck import ready
    assert not ready(state_path)
    warmup.write_state(state_path, {"scheduler_started": False})
    assert not ready(state_path)
    warmup.write_state(state_path, {"scheduler_started": True})
    assert ready(state_path)


def test_heartbeat_updates_timestamp_and_preserves_pending(monkeypatch, state_path):
    old_timestamp = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    state = {"last_checked_at": old_timestamp, "warmup_pending": True}
    state_path.write_text(json.dumps(state))

    async def check():
        updated = asyncio.Event()
        intervals = []

        async def sleep(seconds):
            intervals.append(seconds)
            if len(intervals) == 2:
                updated.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(warmup.asyncio, "sleep", sleep)
        task = asyncio.create_task(warmup.heartbeat(state_path, state))
        await updated.wait()
        current = warmup.read_state(state_path)
        assert current["last_checked_at"] != old_timestamp
        assert current["warmup_pending"]
        assert intervals == [10, 10]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_heartbeat_write_failure_stops_worker(monkeypatch, state_path):
    warmup.write_state(state_path, {"warmup_pending": False})

    async def broken_heartbeat(path, state):
        await asyncio.sleep(0)
        raise OSError("heartbeat write failed")

    monkeypatch.setattr(warmup, "heartbeat", broken_heartbeat)

    async def check():
        monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)
        with pytest.raises(OSError, match="heartbeat write failed"):
            await asyncio.wait_for(worker.main(state_path), timeout=2)
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())
