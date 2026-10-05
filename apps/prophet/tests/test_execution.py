"""Process lifetime and database ownership, including abnormal model exits."""
import multiprocessing
import os
import threading
import time

import pytest

from argus_prophet.scheduling.execution import execute, supervise, before_dispatch, ShutdownRequested


def test_spawn_does_not_inherit_writer_and_is_reaped():
    before = {child.pid for child in multiprocessing.active_children()}
    assert execute(os.getpid) != os.getpid()
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.parametrize('target,args,match', [
    (int, ('invalid',), 'invalid literal'),
    (os._exit, (7,), 'without a result'),
])
def test_failed_process_never_returns_success(target, args, match):
    with pytest.raises(RuntimeError, match=match):
        execute(target, *args)
    assert execute(int, '12') == 12


def test_timeout_terminates_and_reaps_process():
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        execute(time.sleep, 30, timeout_seconds=0.2)
    assert time.monotonic()-started < 5
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_shutdown_drains_active_process_and_prevents_next_dispatch():
    with supervise(grace_seconds=5) as control:
        timer = threading.Timer(0.2, control.stop)
        timer.start()
        try:
            assert execute(time.sleep, 0.5) is None
            with pytest.raises(ShutdownRequested):
                before_dispatch()
        finally:
            timer.join()


def test_shutdown_deadline_terminates_active_process():
    before = {child.pid for child in multiprocessing.active_children()}
    with supervise(grace_seconds=0.1) as control:
        timer = threading.Timer(0.2, control.stop)
        timer.start()
        try:
            with pytest.raises(ShutdownRequested, match='deadline'):
                execute(time.sleep, 30)
        finally:
            timer.join()
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_explicit_control_cancels_executor_thread_process():
    from concurrent.futures import ThreadPoolExecutor
    from argus_prophet.scheduling.execution import ExecutionControl
    before = {child.pid for child in multiprocessing.active_children()}
    control = ExecutionControl()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute, time.sleep, 30, control=control)
        time.sleep(0.2)
        control.cancel()
        with pytest.raises(ShutdownRequested):
            future.result(timeout=5)
    assert {child.pid for child in multiprocessing.active_children()} == before
