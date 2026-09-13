"""Uses the explicitly configured disposable PostgreSQL server, never production."""
from test_domain_storage import database, bootstrap, migrate, runtime
import gzip
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import psycopg
import pytest
from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.observation import Observation
from argus_prophet.ledger import RunRecorder


@pytest.fixture
def recorder_database(database, monkeypatch, tmp_path):
    dsn, passwords, environment = database
    with psycopg.connect(dsn) as conn:
        bootstrap.provision(conn, passwords)
    migrate(environment)
    for key in ('DB_NAME', 'DB_HOST', 'DB_PORT', 'PROPHET_DB_PASSWORD'):
        monkeypatch.setenv(key, environment[key])
    return dsn, passwords, SimpleNamespace(workdir=tmp_path, models_registry={})


@pytest.fixture
def recorder_setup(recorder_database):
    from argus_prophet.worker import generation_lock
    with generation_lock():
        yield recorder_database


def test_snapshot_results_partial_completion_and_role_boundary(recorder_setup, tmp_path):
    dsn, passwords, config = recorder_setup
    run = RunRecorder.begin('all', 'manual', config)
    now = datetime.now(UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    run.snapshot(inputs)
    with pytest.raises(RuntimeError, match='once'):
        run.snapshot(inputs)
    path = tmp_path / 'output.csv'
    content = b'issue_time,value\n2026-09-13T00:00:00Z,4\n'
    path.write_bytes(content)
    run.store('test', path, {'sha256': 'model-hash'}, 1, ['issue_time','value'])
    run.csv_written('test')
    run.skip('density', 'missing source history')
    run.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status,input_sha256,input_snapshot FROM prophet.forecast_run WHERE id=%s', (run.run_id,)).fetchone()[0] == 'partial'
        blob, digest, written = conn.execute("SELECT csv_gzip,sha256,csv_written_at FROM prophet.forecast_artifact WHERE run_id=%s AND name='test'", (run.run_id,)).fetchone()
        assert gzip.decompress(blob) == content and digest == hashlib.sha256(content).hexdigest()
        assert written is not None
        for table in ('clio.measurement', 'api.dashboard_user'):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f'SELECT * FROM {table}')
    for domain in ('api', 'clio'):
        with runtime(dsn, domain, passwords) as conn:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute('SELECT * FROM prophet.forecast_run')


def test_interrupted_and_failed_runs_are_preserved(recorder_setup):
    dsn, passwords, config = recorder_setup
    first = RunRecorder.begin('all', 'manual', config)
    # Simulate recovery performed when a new owner acquires the database lock.
    from argus_prophet.worker import recover_interrupted
    recover_interrupted()
    second = RunRecorder.begin('all', 'manual', config)
    second.finish(error=ValueError('bad model'))
    with runtime(dsn, 'prophet', passwords) as conn:
        rows = dict(conn.execute('SELECT id,status FROM prophet.forecast_run').fetchall())
        assert rows[first.run_id] == 'interrupted'
        assert rows[second.run_id] == 'failed'


def store_product(run, tmp_path, names=('dst_quantile',), issue='2026-09-13T00:00:00Z'):
    from common.schemas.forecast_release import PREDICTION_COLUMNS
    path = tmp_path / 'result.csv'
    for name in names:
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[name]]
        values = [issue, '2026-09-13T01:00:00Z', '1', *('0' for _ in PREDICTION_COLUMNS[name])]
        content = ','.join(columns) + '\n' + ','.join(values) + '\n'
        path.write_text(content)
        run.store(name, path, {}, 1, columns)
    return content


def test_only_completed_runs_publish_and_historical_release_survives(recorder_setup, tmp_path):
    from argus_prophet.publication import read_release, ReleaseNotFound
    _, _, config = recorder_setup
    first = RunRecorder.begin('all', 'manual', config)
    store_product(first, tmp_path)
    with pytest.raises(ReleaseNotFound):
        read_release('dst')
    first.finish()
    old = read_release('dst')
    failed = RunRecorder.begin('all', 'manual', config)
    store_product(failed, tmp_path)
    failed.finish(error=ValueError('other model failed'))
    assert read_release('dst').release_id == old.release_id
    second = RunRecorder.begin('all', 'manual', config)
    store_product(second, tmp_path)
    second.skip('atmospheric_density', 'not ready')
    second.finish()
    assert read_release('dst').run_id == second.run_id
    assert read_release('dst', old.release_id) == old
    # One half of speed is not a publishable product.
    third = RunRecorder.begin('wind', 'manual', config)
    store_product(third, tmp_path, names=('plasma_speed_quantile',))
    third.finish()
    with pytest.raises(ReleaseNotFound):
        read_release('solar-wind-speed')


def test_export_retries_without_recalculation_and_skips_superseded(recorder_setup, tmp_path, monkeypatch):
    from argus_prophet import exports
    from argus_prophet.publication import read_release
    dsn, passwords, config = recorder_setup
    config.models_registry = {'models': {'dst_quantile': {'forecast_path': 'live.csv'}}}
    monkeypatch.setattr(exports, 'get_config', lambda: config)
    run = RunRecorder.begin('all', 'manual', config)
    expected = store_product(run, tmp_path)
    run.finish()
    original_writer = exports.write_csv
    monkeypatch.setattr(exports, 'write_csv', lambda *_, **__: (_ for _ in ()).throw(OSError('disk unavailable')))
    with pytest.raises(RuntimeError, match='pending'):
        exports.export_current()
    assert read_release('dst').run_id == run.run_id
    monkeypatch.setattr(exports, 'write_csv', original_writer)
    assert exports.export_current() == 1
    assert (tmp_path / 'live.csv').read_text() == expected
    assert exports.export_current() == 0
    with runtime(dsn, 'prophet', passwords) as conn:
        attempts, exported, error = conn.execute('SELECT attempts,exported_at,error FROM prophet.forecast_export').fetchone()
        assert attempts == 2 and exported is not None and error is None
    for _ in range(2):
        new = RunRecorder.begin('all', 'manual', config)
        store_product(new, tmp_path)
        new.finish()
    assert exports.export_current() == 1


def test_publication_validation_rolls_back_all_pointers(recorder_setup, tmp_path):
    from argus_prophet.publication import read_release, ReleaseNotFound
    dsn, passwords, config = recorder_setup
    run = RunRecorder.begin('all', 'manual', config)
    store_product(run, tmp_path, names=('plasma_density_quantile',))
    store_product(run, tmp_path, names=('dst_quantile',))
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("UPDATE prophet.forecast_artifact SET sha256=%s WHERE name='dst_quantile'", ('0' * 64,))
    with pytest.raises(ValueError, match='checksum'):
        run.finish()
    with pytest.raises(ReleaseNotFound):
        read_release('solar-wind-density')
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run').fetchone()[0] == 'running'


def test_cutover_is_repeatable_and_never_publishes_failed_attempts(recorder_setup, tmp_path):
    from argus_prophet.publication import publish_existing, read_release
    dsn, passwords, config = recorder_setup
    run = RunRecorder.begin('all', 'manual', config)
    store_product(run, tmp_path)
    # Simulate pre-publication ledger rows created by the previous runtime.
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("UPDATE prophet.forecast_run SET status='succeeded',finished_at=now() WHERE id=%s", (run.run_id,))
    assert publish_existing() == 1
    assert publish_existing() == 0
    assert read_release('dst').run_id == run.run_id
    older = RunRecorder.begin('all', 'manual', config)
    store_product(older, tmp_path, issue='2026-09-12T00:00:00Z')
    older.finish()
    assert read_release('dst').run_id == run.run_id


def test_status_distinguishes_current_release_from_latest_failure(recorder_setup, tmp_path):
    from argus_prophet.readiness import product_status
    _, _, config = recorder_setup
    run = RunRecorder.begin('all', 'manual', config)
    now = datetime(2026, 9, 13, 1, tzinfo=UTC)
    run.snapshot(ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[])))
    store_product(run, tmp_path)
    run.finish()
    failure = RunRecorder.begin('all', 'manual', config)
    failure.finish(error=ValueError('observation service unavailable'))
    status = product_status('dst', now=now)
    assert status.current_release.run_id == run.run_id
    assert status.current_release.input_diagnostics['normalized']['count'] == 0
    assert status.release_age_hours == 1 and status.freshness == 'unconfigured'
    assert status.latest_attempt.run_id == failure.run_id
    assert status.latest_attempt.status == 'failed'
    assert status.latest_attempt_artifacts == []
    unavailable = product_status('hmf', now=now)
    assert unavailable.freshness == 'unavailable' and unavailable.current_release is None
