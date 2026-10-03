from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import URL

from clio.scheduling import observations as scheduler
from clio.db.locks import JOB_LOCKS
from clio.config import ClioObservations

NOW = datetime(2026, 9, 28, 13, 17, tzinfo=UTC)


def config():
    return ClioObservations.model_validate({'observations': {
        metric: {'sources': {'live': ['swpc.rtsw_plasma'], 'historical': ['omni.hourly']},
                 'schedules': {'live': {'every': '1h'}, 'backfill': {'every': every}},
                 'backfill': {'days': 60}}
        for metric, every in [('v', '6h'), ('n', '1d')]}})


@pytest.fixture
def connection(monkeypatch):
    conn = Mock()
    state = {'locked': True, 'markers': {}, 'writes': [], 'locks': []}

    def execute(sql, params):
        if 'pg_try_advisory_lock' in sql:
            state['locks'].append(params[0])
            assert params[0] in (JOB_LOCKS['refresh'], JOB_LOCKS['live'], JOB_LOCKS['aia'], JOB_LOCKS['aia-live'])
            return Mock(fetchone=lambda: (state['locked'],))
        if sql.startswith('SELECT name'):
            return Mock(fetchall=lambda: list(state['markers'].items()))
        if sql.startswith('DELETE'):
            for name in params[0]:
                state['markers'].pop(name, None)
            return
        assert sql.strip().startswith('INSERT INTO clio.scheduled_job')
        state['markers'][params[0]] = params[1]
        state['writes'].append(params)

    conn.execute.side_effect = execute
    conn.transaction.side_effect = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, 'psycopg', SimpleNamespace(connect=lambda *args, **kwargs: nullcontext(conn)))
    monkeypatch.setattr(scheduler, 'get_database_url', lambda: URL.create('postgresql', database='test'))
    return state


def test_batches_due_observations_and_skips_completed_slots_on_restart(connection):
    run = Mock(return_value={'failed_metrics': []})
    assert scheduler.execute(config(), run, now=NOW)
    run.assert_called_once_with(['v', 'n'], NOW)
    assert connection['markers'] == {
        'backfill.v': NOW.replace(hour=12, minute=0), 'backfill.n': NOW.replace(hour=0, minute=0)}
    run.reset_mock()
    assert not scheduler.execute(config(), run, now=NOW + timedelta(minutes=30))
    run.assert_not_called()
    assert scheduler.execute(config(), run, now=NOW.replace(hour=18))
    assert run.call_args.args[0] == ['v']


def test_failed_metric_retries_without_redownloading_successful_metric(connection):
    run = Mock(return_value={'failed_metrics': ['n']})
    scheduler.execute(config(), run, now=NOW)
    assert set(connection['markers']) == {'backfill.v'}
    run.return_value = {'failed_metrics': []}
    scheduler.execute(config(), run, now=NOW + timedelta(minutes=1))
    assert run.call_args.args[0] == ['n']
    assert set(connection['markers']) == {'backfill.v', 'backfill.n'}


def test_lock_contention_and_batch_failure_never_mark_complete(connection):
    run = Mock(side_effect=RuntimeError('database failed'))
    connection['locked'] = False
    with pytest.raises(scheduler.JobBusy):
        scheduler.execute(config(), run, now=NOW)
    run.assert_not_called()
    connection['locked'] = True
    with pytest.raises(RuntimeError, match='database failed'):
        scheduler.execute(config(), run, now=NOW)
    assert not connection['markers']


def test_slots_use_utc_for_long_and_non_hour_intervals():
    assert scheduler.observation_slot(NOW, timedelta(hours=6)) == NOW.replace(hour=12, minute=0)
    assert scheduler.observation_slot(NOW, timedelta(days=1)) == NOW.replace(hour=0, minute=0)
    assert scheduler.observation_slot(NOW, timedelta(minutes=90)) == NOW.replace(hour=12, minute=0)
    assert scheduler.observation_slot(NOW.astimezone(), timedelta(hours=6)) == NOW.replace(hour=12, minute=0)


def test_empty_config_does_not_connect_or_run(monkeypatch):
    connect, run = Mock(), Mock()
    monkeypatch.setitem(sys.modules, 'psycopg', SimpleNamespace(connect=connect))
    assert not scheduler.execute(ClioObservations(observations={}), run)
    connect.assert_not_called()
    run.assert_not_called()


def test_polling_honors_subminute_configured_intervals(monkeypatch):
    from clio.config import Schedule
    cfg = config()
    cfg.observations['v'].schedules.backfill = Schedule(every='30s')
    from clio.worker import observation_task
    task = observation_task(cfg, 'backfill', 'numeric', {}, Mock())
    assert task.interval == 30


def test_live_and_backfill_have_independent_markers(connection):
    run = Mock(return_value={'failed_metrics': []})
    scheduler.execute(config(), run, now=NOW, mode='backfill')
    scheduler.execute(config(), run, now=NOW, mode='live')
    assert set(connection['markers']) == {'live.v', 'live.n', 'backfill.v', 'backfill.n'}
    assert connection['markers']['live.v'] == NOW.replace(minute=0)


def test_forced_live_restart_retries_failure_even_if_previous_process_completed_slot(connection):
    run = Mock(return_value={'failed_metrics': []})
    scheduler.execute(config(), run, now=NOW, mode='live')
    run.return_value = {'failed_metrics': ['n']}
    scheduler.execute(config(), run, now=NOW, mode='live', force=True)
    assert 'live.n' not in connection['markers']
    run.return_value = {'failed_metrics': []}
    scheduler.execute(config(), run, now=NOW, mode='live')
    assert run.call_args.args[0] == ['n']


def test_file_schedules_have_separate_locks_and_common_metric_markers(connection):
    from clio.config import load_observation_config
    full = load_observation_config()
    cfg = full.model_copy(update={'observations': {'aia193': full.observations['aia193']}})
    run = Mock(return_value={'failed_metrics': []})
    scheduler.execute(cfg, run, now=NOW, mode='live')
    scheduler.execute(cfg, run, now=NOW, mode='backfill')
    assert connection['locks'] == [JOB_LOCKS['aia-live'], JOB_LOCKS['aia']]
    assert set(connection['markers']) == {'live.aia193', 'backfill.aia193'}
