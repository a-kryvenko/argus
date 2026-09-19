"""Worker lifecycle must not leave a partially running service or orphan writers."""
import asyncio
import os
import signal
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest

from argus_clio import worker
from argus_clio.commands import collect_solar_wind, collect_geomagnetic
from argus_clio.commands._shutdown import stop_on_signal, wait_for_next_poll
from argus_clio.services.collector_heartbeat import check_heartbeat


@pytest.fixture
def lifecycle(monkeypatch):
    stopped = Mock()
    stopped.is_set.return_value = False
    stopped.wait.side_effect = [False, True]
    monkeypatch.setattr(worker.threading, 'Event', lambda: stopped)
    handlers = {}

    def install(sig, handler):
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(worker.signal, 'signal', install)
    children = [Mock() for _ in worker.COMMANDS]
    for child in children:
        child.poll.return_value = None
    launch = Mock(side_effect=children)
    monkeypatch.setattr(worker.subprocess, 'Popen', launch)
    return children, launch, handlers


def test_worker_starts_independent_loops_and_drains_all_on_stop(lifecycle):
    children, launch, handlers = lifecycle
    worker.work()
    assert launch.call_count == 4
    assert [call.args[0][3:] for call in launch.call_args_list] == [list(c) for c in worker.COMMANDS]
    for child in children:
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
    assert set(handlers.values()) == {signal.SIG_DFL}


@pytest.mark.parametrize('exit_code', [0, 1])
def test_even_clean_child_exit_fails_service_and_stops_other_writers(lifecycle, exit_code):
    children, _, handlers = lifecycle
    children[0].poll.return_value = exit_code
    with pytest.raises(RuntimeError, match='exited unexpectedly'):
        worker.work()
    children[0].terminate.assert_not_called()
    for child in children[1:]:
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
    assert set(handlers.values()) == {signal.SIG_DFL}


def test_partial_startup_failure_drains_started_child(lifecycle):
    children, launch, _ = lifecycle
    launch.side_effect = [children[0], OSError('cannot spawn')]
    with pytest.raises(OSError, match='cannot spawn'):
        worker.work()
    children[0].terminate.assert_called_once()
    children[0].wait.assert_called_once()


def test_shutdown_signals_everyone_before_waiting_and_kills_at_deadline(monkeypatch):
    children = [Mock(), Mock()]
    for child in children:
        child.poll.return_value = None
    monkeypatch.setattr(worker, 'STOP_TIMEOUT_SECONDS', 0)

    def wait(**kwargs):
        for child in children:
            child.terminate.assert_called_once()
        if 'timeout' in kwargs:
            assert kwargs['timeout'] == 0
            raise subprocess.TimeoutExpired('collector', 0)

    for child in children:
        child.wait.side_effect = wait
    worker.stop_children(children)
    for child in children:
        child.kill.assert_called_once()
        assert child.wait.call_count == 2


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
    monkeypatch.setattr(collect_solar_wind, 'dispose_engine', dispose)
    previous = signal.getsignal(signal.SIGTERM)
    asyncio.run(asyncio.wait_for(collect_solar_wind.collect(True), timeout=2))
    assert events == ['started', 'saved']
    dispose.assert_awaited_once()
    assert check_heartbeat('solar-wind')['reason'] == 'collector_stopped'
    assert signal.getsignal(signal.SIGTERM) == previous


def test_geomagnetic_finishes_active_poll_without_starting_another(monkeypatch):
    async def scenario():
        stopped = asyncio.Event()

        async def ingest(metric):
            stopped.set()
            await asyncio.sleep(0)

        ingest_mock = AsyncMock(side_effect=ingest)
        monkeypatch.setattr(collect_geomagnetic, 'ingest_source', ingest_mock)
        await asyncio.wait_for(collect_geomagnetic.watch_source('dst', stopped=stopped), timeout=2)
        ingest_mock.assert_awaited_once_with('dst')

    asyncio.run(scenario())


def test_signal_wakes_idle_collector_without_waiting_for_poll_interval():
    async def scenario():
        with stop_on_signal(True) as stopped:
            asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
            await asyncio.wait_for(wait_for_next_poll(stopped, 300), timeout=2)
            assert stopped.is_set()

    asyncio.run(scenario())
