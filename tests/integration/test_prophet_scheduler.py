"""Scheduling, publication atomicity and session-loss tests on disposable PostgreSQL."""
from test_prophet_ledger import recorder_database, database, store_product
from test_domain_storage import runtime
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import psycopg
import pytest

from argus_prophet.db.session import require_writer
from argus_prophet.ledger import RunRecorder
from argus_prophet.publication import read_release, ReleaseNotFound
from argus_prophet.worker import generation_lock, run_due, due_slot, import_schedule, GenerationBusy

NOW = datetime(2026, 9, 13, 10, 10, tzinfo=UTC)


def generate(config, tmp_path, *, partial=False):
    def calculate(slot):
        run = RunRecorder.begin('all', 'scheduled', config, scheduled_slot=slot)
        store_product(run, tmp_path)
        if partial:
            run.skip('atmospheric_density', 'history unavailable')
        run.finish()
    return calculate


def test_restart_partial_completion_and_manual_runs_do_not_consume_slots(recorder_database, tmp_path):
    dsn, passwords, config = recorder_database
    with generation_lock():
        manual = RunRecorder.begin('all', 'manual', config)
        store_product(manual, tmp_path)
        manual.finish()
    assert run_due(generate(config, tmp_path, partial=True), NOW)
    assert not run_due(lambda _: pytest.fail('Completed slot must not repeat'), NOW)
    assert run_due(generate(config, tmp_path), NOW + timedelta(hours=3))
    with runtime(dsn, 'prophet', passwords) as conn:
        rows = conn.execute('SELECT slot,status,attempts FROM prophet.forecast_slot ORDER BY slot').fetchall()
        assert rows == [(due_slot(NOW), 'partial', 1), (due_slot(NOW) + timedelta(hours=3), 'succeeded', 1)]
        assert conn.execute('SELECT scheduled_slot FROM prophet.forecast_run WHERE id=%s', (manual.run_id,)).fetchone()[0] is None


def test_failed_slot_retries_and_keeps_attempt_history(recorder_database, tmp_path):
    dsn, passwords, config = recorder_database
    def fail(slot):
        run = RunRecorder.begin('all', 'scheduled', config, scheduled_slot=slot)
        error = ValueError('model failure')
        run.finish(error=error)
        raise error
    with pytest.raises(ValueError):
        run_due(fail, NOW)
    assert run_due(generate(config, tmp_path), NOW)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status,attempts,error FROM prophet.forecast_slot').fetchone() == ('succeeded', 2, None)
        assert conn.execute('SELECT status FROM prophet.forecast_run ORDER BY started_at').fetchall() == [('failed',), ('succeeded',)]


def test_crash_after_publication_does_not_repeat_but_unfinished_attempt_recovers(recorder_database, tmp_path):
    dsn, passwords, config = recorder_database
    def committed_then_crashed(slot):
        generate(config, tmp_path)(slot)
        raise RuntimeError('process crashed after commit')
    with pytest.raises(RuntimeError):
        run_due(committed_then_crashed, NOW)
    assert not run_due(lambda _: pytest.fail('Committed slot must survive restart'), NOW)
    def unfinished(slot):
        RunRecorder.begin('all', 'scheduled', config, scheduled_slot=slot)
        raise RuntimeError('process crashed before completion')
    with pytest.raises(RuntimeError):
        run_due(unfinished, NOW + timedelta(hours=1))
    # Recovery also works when deployment changes the workdir; scope is audit metadata.
    other = SimpleNamespace(workdir=tmp_path / 'another-workdir', models_registry={})
    assert run_due(generate(other, tmp_path), NOW + timedelta(hours=1))
    with runtime(dsn, 'prophet', passwords) as conn:
        statuses = conn.execute('SELECT status FROM prophet.forecast_run ORDER BY started_at').fetchall()
        assert statuses == [('succeeded',), ('interrupted',), ('succeeded',)]
        assert conn.execute('SELECT attempts FROM prophet.forecast_slot ORDER BY slot DESC LIMIT 1').fetchone()[0] == 2


def test_independent_sessions_cannot_generate_or_export_concurrently(recorder_database):
    with generation_lock(), ThreadPoolExecutor(max_workers=1) as pool:
        def contender():
            with generation_lock():
                pytest.fail('Competing writer acquired the lock')
        with pytest.raises(GenerationBusy):
            pool.submit(contender).result(timeout=10)
    # Normal release makes the same database immediately available again.
    with generation_lock():
        assert require_writer().execute('SELECT 1').fetchone()[0] == 1


def test_disconnected_writer_cannot_publish_after_new_owner_takes_over(recorder_database, tmp_path):
    dsn, passwords, config = recorder_database
    with generation_lock():
        old = RunRecorder.begin('all', 'scheduled', config, scheduled_slot=due_slot(NOW))
        store_product(old, tmp_path)
        pid = require_writer().info.backend_pid
        with psycopg.connect(dsn, autocommit=True) as admin:
            assert admin.execute('SELECT pg_terminate_backend(%s)', (pid,)).fetchone()[0]
        with pytest.raises((psycopg.Error, RuntimeError)):
            old.finish()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(run_due, generate(config, tmp_path), NOW).result(timeout=20)
        # The old context stays bound to its broken session, never a new connection.
        with pytest.raises((psycopg.Error, RuntimeError)):
            old.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (old.run_id,)).fetchone()[0] == 'interrupted'
        assert conn.execute('SELECT count(*) FROM prophet.forecast_release').fetchone()[0] == 1
    assert read_release('dst').run_id != old.run_id


def test_slot_completion_and_publication_roll_back_together(recorder_database, tmp_path):
    dsn, passwords, config = recorder_database
    with generation_lock():
        run = RunRecorder.begin('all', 'scheduled', config, scheduled_slot=due_slot(NOW))
        store_product(run, tmp_path)
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute("UPDATE prophet.forecast_artifact SET sha256=%s", ('0' * 64,))
        with pytest.raises(ValueError, match='checksum'):
            run.finish()
        with pytest.raises(ReleaseNotFound):
            read_release('dst')
        with runtime(dsn, 'prophet', passwords) as conn:
            assert conn.execute('SELECT status FROM prophet.forecast_slot').fetchone()[0] == 'running'
            assert conn.execute('SELECT status FROM prophet.forecast_run').fetchone()[0] == 'running'
    assert run_due(generate(config, tmp_path), NOW)


def test_export_failure_does_not_reopen_completed_slot(recorder_database, tmp_path, monkeypatch):
    from argus_prophet import exports
    _, _, config = recorder_database
    config.models_registry = {'models': {'dst_quantile': {'forecast_path': 'live.csv'}}}
    monkeypatch.setattr(exports, 'get_config', lambda: config)
    assert run_due(generate(config, tmp_path), NOW)
    original = exports.write_csv
    def fail(*_, **__):
        raise OSError('disk full')
    monkeypatch.setattr(exports, 'write_csv', fail)
    with generation_lock(), pytest.raises(RuntimeError, match='pending'):
        exports.export_current()
    assert not run_due(lambda _: pytest.fail('Export failure must not recalculate'), NOW)
    monkeypatch.setattr(exports, 'write_csv', original)
    with generation_lock():
        assert exports.export_current() == 1
    assert (tmp_path / 'live.csv').is_file()


def test_marker_import_is_idempotent_and_does_not_fabricate_run_history(recorder_database, tmp_path):
    dsn, passwords, _ = recorder_database
    marker = tmp_path / 'last-completed-slot'
    content = due_slot(NOW).isoformat() + '\n'
    marker.write_text(content)
    with generation_lock():
        assert import_schedule(marker, now=NOW)
        assert not import_schedule(marker, now=NOW)
    assert not run_due(lambda _: pytest.fail('Imported completion must skip the slot'), NOW)
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status,attempts FROM prophet.forecast_slot').fetchone() == ('imported', 0)
        assert conn.execute('SELECT count(*) FROM prophet.forecast_run').fetchone()[0] == 0
    assert marker.read_text() == content
