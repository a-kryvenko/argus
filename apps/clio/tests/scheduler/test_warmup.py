import asyncio
import json
import signal
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from clio.scheduler import warmup, worker


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "worker-state.json"


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


async def run_until_scheduler_starts(monkeypatch, state_path):
    class Scheduler(AsyncIOScheduler):
        def start(self, *args, **kwargs):
            super().start(*args, **kwargs)
            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)

    monkeypatch.setattr(worker, "AsyncIOScheduler", Scheduler)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)
    await worker.main(state_path)


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


def test_cold_start_backfills_before_collect_and_persists_state(monkeypatch, state_path):
    calls = []
    events = []

    async def backfill():
        assert warmup.read_state(state_path)["warmup_pending"]
        calls.append("backfill")
        return {"status": "complete"}

    async def collect(metrics):
        calls.append(metrics)
        return {"status": "complete"}

    monkeypatch.setattr(warmup, "run_backfill", backfill)
    monkeypatch.setattr(warmup, "run_collect", collect)
    monkeypatch.setattr(worker, "collecting_schedules", lambda config: [(["v"], 1), (["dst"], 5)])
    monkeypatch.setattr(worker, "on_job_finished", events.append)

    async def check():
        await run_until_scheduler_starts(monkeypatch, state_path)
        assert calls == ["backfill", ["v"], ["dst"]]
        assert not warmup.read_state(state_path)["warmup_pending"]
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())
    assert len(events) == 3
    assert all(event.code == EVENT_JOB_EXECUTED for event in events)
    assert warmup.read_state(state_path)["last_checked_at"]
    assert not state_path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("seconds,expected_calls", [(10, 0), (120, 1)])
def test_restart_checks_previous_heartbeat(monkeypatch, state_path, seconds, expected_calls):
    state_path.write_text(json.dumps({
        "last_checked_at": (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(),
        "warmup_pending": False,
    }))
    backfill = AsyncMock(return_value={"status": "complete"})
    collect = AsyncMock(return_value={"status": "complete"})
    monkeypatch.setattr(warmup, "run_backfill", backfill)
    monkeypatch.setattr(warmup, "run_collect", collect)
    monkeypatch.setattr(worker, "collecting_schedules", lambda config: [(["v"], 1)])
    monkeypatch.setattr(worker, "on_job_finished", lambda event: None)

    async def check():
        await run_until_scheduler_starts(monkeypatch, state_path)

    asyncio.run(check())
    assert backfill.await_count == collect.await_count == expected_calls


def test_backfill_failure_is_reported_and_collect_continues(monkeypatch, state_path):
    error = RuntimeError("source unavailable")
    monkeypatch.setattr(warmup, "run_backfill", AsyncMock(side_effect=error))
    collect = AsyncMock(return_value={"status": "partial", "failed_metrics": ["v"]})
    monkeypatch.setattr(warmup, "run_collect", collect)
    events = []
    monkeypatch.setattr(worker, "collecting_schedules", lambda config: [(["v"], 1)])
    monkeypatch.setattr(worker, "on_job_finished", events.append)

    async def check():
        await run_until_scheduler_starts(monkeypatch, state_path)

    asyncio.run(check())
    assert events[0].code == EVENT_JOB_ERROR
    assert events[0].exception is error
    assert events[1].retval["status"] == "partial"
    collect.assert_awaited_once_with(["v"])
    assert not warmup.read_state(state_path)["warmup_pending"]


@pytest.mark.parametrize("during_warmup", [True, False])
def test_worker_signal_stops_during_warmup_or_scheduler(monkeypatch, state_path, during_warmup):
    ready = None
    signals = {}
    schedulers = []
    backfill_cancelled = []
    events = []

    async def backfill():
        if during_warmup:
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                backfill_cancelled.append(True)
        return {"status": "complete"}

    class Scheduler(AsyncIOScheduler):
        def start(self, *args, **kwargs):
            super().start(*args, **kwargs)
            assert not warmup.read_state(state_path)["warmup_pending"]
            assert len(self.get_jobs()) == len(worker.collecting_schedules(worker.load_observation_config())) + 3
            ready.set()

    def create_scheduler(**kwargs):
        scheduler = Scheduler(**kwargs)
        schedulers.append(scheduler)
        return scheduler

    monkeypatch.setattr(warmup, "run_backfill", backfill)
    collect = AsyncMock(return_value={"status": "complete"})
    monkeypatch.setattr(warmup, "run_collect", collect)
    monkeypatch.setattr(worker, "on_job_finished", events.append)
    monkeypatch.setattr(worker, "AsyncIOScheduler", create_scheduler)

    async def check():
        nonlocal ready
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: signals.update({sig: callback}))
        task = asyncio.create_task(worker.main(state_path))
        await asyncio.wait_for(ready.wait(), timeout=2)
        signals[signal.SIGTERM]()
        await asyncio.wait_for(task, timeout=2)
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())
    assert not schedulers[0].running
    assert warmup.read_state(state_path)["warmup_pending"] is during_warmup
    if during_warmup:
        assert backfill_cancelled == [True]
        collect.assert_not_awaited()
        assert events == []
    else:
        assert collect.await_count == len(worker.collecting_schedules(worker.load_observation_config()))
        assert [call.args[0] for call in collect.await_args_list] == [
            metrics for metrics, _ in worker.collecting_schedules(worker.load_observation_config())
        ]


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
