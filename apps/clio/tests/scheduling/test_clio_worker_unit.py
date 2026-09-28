"""One scheduler bounds concurrency, retries failures and drains on shutdown."""
import asyncio
from concurrent.futures import Future
import os
import signal
from unittest.mock import AsyncMock, Mock

import pytest

from clio import worker
from clio.commands import collect_solar_wind
from clio.commands._shutdown import stop_on_signal, wait_for_next_poll
from clio.monitoring.heartbeat import check_heartbeat


def test_live_lanes_are_not_queued_behind_background_jobs(monkeypatch):
    monkeypatch.setattr(worker, 'monotonic', lambda: 100)
    tasks = [worker.Task(f'backfill-{i}', Mock(), background=True) for i in range(4)]
    tasks += [worker.Task(name, Mock()) for name in ('native-wind', 'live-numeric', 'live-file')]
    pool = Mock()
    pool.submit.side_effect = lambda run: Future()
    running = {}
    worker.launch_due(tasks, running, pool)
    assert set(running) == {'backfill-0', 'backfill-1', 'native-wind', 'live-numeric', 'live-file'}
    worker.launch_due(tasks, running, pool)
    assert pool.submit.call_count == 5


def test_completed_and_failed_jobs_are_retried_without_starving_waiting_jobs(monkeypatch):
    monkeypatch.setattr(worker, 'monotonic', lambda: 100)
    tasks = [worker.Task(f'backfill-{i}', Mock(), background=True) for i in range(3)]
    good, bad = Future(), Future()
    good.set_result(True)
    bad.set_exception(RuntimeError('source unavailable'))
    running = {tasks[0].name: (tasks[0], good, 0), tasks[1].name: (tasks[1], bad, 0)}
    worker.finish_tasks(running)
    assert not running
    assert tasks[0].next_run == tasks[1].next_run == 160
    pool = Mock()
    pool.submit.side_effect = lambda run: Future()
    worker.launch_due(tasks, running, pool)
    assert list(running) == ['backfill-2']


def test_worker_stops_launching_and_drains_active_work(monkeypatch):
    events = []
    beat = Mock()
    monkeypatch.setattr(worker, 'CollectorHeartbeat', lambda _: beat)
    monkeypatch.setattr(worker, 'collector_sources', lambda _: [])
    monkeypatch.setattr(worker, 'load_observation_config', lambda: object())
    def tasks(config, beats, abort):
        def task():
            events.append('started')
            os.kill(os.getpid(), signal.SIGTERM)
            import time
            time.sleep(0.1)
            assert not abort.is_set()
            events.append('finished')
            return True
        return [worker.Task('native-wind', task)]
    monkeypatch.setattr(worker, 'tasks_for', tasks)
    previous = signal.getsignal(signal.SIGTERM)
    worker.work()
    assert events == ['started', 'finished']
    beat.stop.assert_called_once()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_shutdown_deadline_cancels_executors(monkeypatch):
    beat = Mock()
    monkeypatch.setattr(worker, 'CollectorHeartbeat', lambda _: beat)
    monkeypatch.setattr(worker, 'collector_sources', lambda _: [])
    monkeypatch.setattr(worker, 'load_observation_config', lambda: object())
    monkeypatch.setattr(worker, 'STOP_TIMEOUT_SECONDS', 0)
    def tasks(config, beats, abort):
        def task():
            os.kill(os.getpid(), signal.SIGTERM)
            assert abort.wait(2)
        return [worker.Task('native-wind', task)]
    monkeypatch.setattr(worker, 'tasks_for', tasks)
    worker.work()
    beat.stop.assert_called_once()


def test_observation_lane_preserves_partial_report_time_and_heartbeat(monkeypatch):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace
    now = datetime.now(UTC)
    config = SimpleNamespace(observations={'kp': SimpleNamespace(
        schedules=SimpleNamespace(live=SimpleNamespace(every=timedelta(seconds=60))))})
    beat = Mock(sources={'kp': {}})
    result = {'failed_metrics': ['kp']}
    invoke = Mock(return_value=result)
    forces = []
    def execute(config, run, **kwargs):
        forces.append(kwargs['force'])
        assert run(['kp'], now) is result
        return True
    monkeypatch.setattr(worker.observations, 'execute', execute)
    task = worker.observation_task(config, 'live', 'numeric', {'geomagnetic': beat}, invoke)
    task.run()
    task.run()
    assert forces == [True, False]
    assert invoke.call_args.kwargs['now'] == now
    assert invoke.call_args.kwargs['heartbeat'] is None
    assert beat.started.call_count == beat.finished.call_count == 2


def test_solar_wind_finishes_active_collection_on_sigterm(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_COLLECTOR_HEALTH_DIR', str(tmp_path))
    events = []

    async def refresh():
        events.append('started')
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0)
        events.append('saved')

    monkeypatch.setattr(collect_solar_wind, 'refresh_solar_wind', refresh)
    dispose = AsyncMock()
    from clio import cli
    from clio.db import session
    from types import SimpleNamespace
    monkeypatch.setattr(session, 'dispose_engine', dispose)
    previous = signal.getsignal(signal.SIGTERM)
    cli.invoke('native-wind', SimpleNamespace(watch=True))
    assert events == ['started', 'saved']
    dispose.assert_awaited_once()
    assert check_heartbeat('solar-wind')['reason'] == 'collector_stopped'
    assert signal.getsignal(signal.SIGTERM) == previous




def test_signal_wakes_idle_collector_without_waiting_for_poll_interval():
    async def scenario():
        with stop_on_signal(True) as stopped:
            asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
            await asyncio.wait_for(wait_for_next_poll(stopped, 300), timeout=2)
            assert stopped.is_set()

    asyncio.run(scenario())
