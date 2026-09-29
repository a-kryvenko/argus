from concurrent.futures import Future
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus_prophet import worker
from argus_prophet.config import ProductSchedule, ProphetConfig
from argus_prophet.scheduling.execution import ExecutionControl, ShutdownRequested

NOW = datetime(2026, 9, 29, 10, 10, tzinfo=UTC)


def test_independent_schedule_boundaries_and_timezones():
    hourly, frequent = ProductSchedule(), ProductSchedule(every_minutes=5, offset_minutes=0)
    before = NOW.replace(minute=9)
    assert hourly.due_slot(before) == before.replace(hour=9, minute=0)
    assert hourly.due_slot(NOW) == NOW.replace(minute=0)
    assert frequent.due_slot(before) == NOW.replace(minute=5)
    assert frequent.due_slot(NOW) == NOW
    assert hourly.due_slot(NOW.astimezone(timezone(timedelta(hours=2)))) == hourly.due_slot(NOW)
    with pytest.raises(ValueError, match='timezone'):
        hourly.due_slot(NOW.replace(tzinfo=None))


@pytest.fixture
def dispatcher(monkeypatch):
    events, submitted, runs = [], [], {}
    @contextmanager
    def lock():
        events.append('lock')
        try:
            yield
        finally:
            events.append('unlock')
    def begin(product, *args, **kwargs):
        run = Mock(run_id=product)
        run.snapshot.side_effect = lambda _: events.append(('snapshot', product))
        run.finish.side_effect = lambda **_: events.append(('finish', product))
        runs[product] = run
        return run
    def submit(target, *args, **kwargs):
        future = Future()
        submitted.append((target, args, future))
        events.append(('submit', args[0].__name__))
        return future
    monkeypatch.setattr(worker, 'generation_lock', lock)
    monkeypatch.setattr(worker, 'product_pending', lambda *_: True)
    monkeypatch.setattr(worker, 'provenance', lambda _: {})
    monkeypatch.setattr(worker.RunRecorder, 'begin', begin)
    pool = Mock(submit=submit)
    dispatcher = worker.Dispatcher(ProphetConfig(), SimpleNamespace(workdir='models', models_registry={'models': {}}),
                                   ExecutionControl(), pool)
    dispatcher.tasks = [worker.Task('dst', ProductSchedule()),
                        worker.Task('hmf', ProductSchedule(every_minutes=5, offset_minutes=0))]
    return dispatcher, submitted, runs, events


def test_shared_inputs_precede_calculations_and_fast_product_publishes_independently(dispatcher):
    d, submitted, runs, events = dispatcher
    d.launch_due(NOW, 0)
    assert len(submitted) == 1 and len(d.active) == 2
    assert events.count('lock') == 1
    inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    submitted[0][2].set_result(inputs)
    d.finish_tasks()
    assert len(submitted) == 3
    assert events.index(('snapshot', 'dst')) < events.index(('submit', 'calculate_product'))
    runs['dst'].snapshot.assert_called_once_with(inputs)
    runs['hmf'].snapshot.assert_called_once_with(inputs)
    # HMF finishes while the slow Dst calculation still owns a process.
    submitted[2][2].set_result([])
    d.finish_tasks()
    runs['hmf'].finish.assert_called_once_with()
    runs['dst'].finish.assert_not_called()
    assert 'unlock' not in events
    d.launch_due(NOW + timedelta(minutes=5), 300)
    assert len(submitted) == 4 and len(d.active) == 2
    assert d.active['dst'].future is submitted[1][2]


def test_capacity_failure_retry_and_shutdown_do_not_overlap_products(dispatcher):
    d, submitted, runs, events = dispatcher
    d.config.max_parallel_products = 1
    d.launch_due(NOW, 0)
    assert set(d.active) == {'dst'}
    submitted[0][2].set_exception(TimeoutError('input timeout'))
    d.finish_tasks()
    assert isinstance(runs['dst'].finish.call_args.kwargs['error'], TimeoutError)
    assert events[-1] == 'unlock'
    d.launch_due(NOW, 1)
    assert set(d.active) == {'hmf'}  # older pending task goes first
    d.control.stop()
    submitted[1][2].set_result(object())
    d.finish_tasks()
    assert isinstance(runs['hmf'].finish.call_args.kwargs['error'], ShutdownRequested)
    assert len(submitted) == 2  # shutdown never starts a new calculation
    with pytest.raises(ShutdownRequested):
        d.launch_due(NOW, 60)
    assert not d.active and not d.locked


def test_verification_dispatches_while_all_product_slots_are_busy(dispatcher):
    d, submitted, runs, events = dispatcher
    d.config.verification.enabled = True
    d.launch_due(NOW, 0)
    assert len(d.active) == 2
    assert submitted[-1][1][0] is worker.verify_due
    assert d.verification is submitted[-1][2]
    d.launch_due(NOW, 1)
    assert len(submitted) == 2


def test_failed_writer_cancels_children_before_releasing_lock(dispatcher):
    d, submitted, runs, events = dispatcher
    d.launch_due(NOW, 0)
    submitted[0][2].set_exception(RuntimeError('failed input'))
    runs['dst'].finish.side_effect = ConnectionError('lost writer')
    with pytest.raises(ConnectionError):
        d.finish_tasks()
    d.pool.shutdown.side_effect = lambda **_: events.append('reaped')
    d.close()
    assert d.control.stopped.is_set()
    assert events[-2:] == ['reaped', 'unlock']


def test_shutdown_publishes_already_running_calculations(dispatcher):
    d, submitted, runs, events = dispatcher
    d.launch_due(NOW, 0)
    submitted[0][2].set_result(SimpleNamespace(observations=SimpleNamespace(points=[])))
    d.finish_tasks()
    d.control.stop()
    for _, _, future in submitted[1:]:
        future.set_result([])
    d.finish_tasks()
    assert not d.active and events[-1] == 'unlock'
    for run in runs.values():
        run.finish.assert_called_once_with()
