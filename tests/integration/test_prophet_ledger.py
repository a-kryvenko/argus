"""Uses the explicitly configured disposable PostgreSQL server, never production."""
from test_domain_storage import database, migrate, runtime
import gzip
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import psycopg
import pytest
from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.observation import Observation
from argus_prophet.services.runs import RunRecorder, describe_run
from argus_prophet.services.releases.publication import read_release, ReleaseNotFound


@pytest.fixture
def recorder_database(database, monkeypatch, tmp_path):
    dsn, passwords, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('PROPHET_DB_'):
            monkeypatch.setenv(key, value)
    return dsn, passwords, SimpleNamespace(workdir=tmp_path, models_registry={})


@pytest.fixture
def recorder_setup(recorder_database):
    from argus_prophet.worker import generation_lock
    with generation_lock():
        yield recorder_database


def store_product(run, names=('dst_quantile',), issue='2026-09-13T00:00:00+00:00'):
    from common.schemas.forecast_release import PREDICTION_COLUMNS
    valid = (datetime.fromisoformat(issue) + timedelta(hours=1)).isoformat()
    for name in names:
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[name]]
        values = [issue, valid, '1', *('0' for _ in PREDICTION_COLUMNS[name])]
        content = ','.join(columns) + '\n' + ','.join(values) + '\n'
        run.store(name, content.encode('utf-8'), {}, 1, columns)
    return content


def test_snapshot_bytes_publication_and_role_boundary(recorder_setup):
    dsn, passwords, config = recorder_setup
    run = RunRecorder.begin('dst', 'manual', config)
    now = datetime.now(UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    run.snapshot(inputs)
    with pytest.raises(RuntimeError, match='once'):
        run.snapshot(inputs)
    content = store_product(run).encode('utf-8')
    run.finish()
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (run.run_id,)).fetchone()[0] == 'succeeded'
        blob, digest = conn.execute('SELECT csv_gzip,sha256 FROM prophet.forecast_artifact WHERE run_id=%s', (run.run_id,)).fetchone()
        assert gzip.decompress(blob) == content and digest == hashlib.sha256(content).hexdigest()
        assert conn.execute("SELECT to_regclass('prophet.forecast_export')").fetchone()[0] is None
        assert not conn.execute("SELECT 1 FROM information_schema.columns WHERE table_schema='prophet' AND table_name='forecast_artifact' AND column_name='csv_written_at'").fetchone()
        for table in ('clio.measurement', 'api.dashboard_user'):
            with pytest.raises(psycopg.errors.UndefinedTable):
                conn.execute(f'SELECT * FROM {table}')
    for domain in ('api', 'clio'):
        with runtime(dsn, domain, passwords) as conn:
            with pytest.raises(psycopg.errors.UndefinedTable):
                conn.execute('SELECT * FROM prophet.forecast_run')
    details = describe_run(run.run_id)
    assert len(details['releases']) == 1
    assert 'csv_written_at' not in details['artifacts'][0]
    assert read_release('dst').artifacts[0].csv_text.encode('utf-8') == content


def test_interrupted_and_failed_runs_are_preserved(recorder_setup):
    dsn, passwords, config = recorder_setup
    first = RunRecorder.begin('dst', 'manual', config)
    from argus_prophet.worker import recover_interrupted
    recover_interrupted()
    second = RunRecorder.begin('dst', 'manual', config)
    second.finish(error=ValueError('bad model'))
    with runtime(dsn, 'prophet', passwords) as conn:
        rows = dict(conn.execute('SELECT id,status FROM prophet.forecast_run').fetchall())
        assert rows[first.run_id] == 'interrupted'
        assert rows[second.run_id] == 'failed'


def test_only_completed_products_publish_and_history_survives(recorder_setup):
    _, _, config = recorder_setup
    first = RunRecorder.begin('dst', 'manual', config)
    store_product(first)
    with pytest.raises(ReleaseNotFound):
        read_release('dst')
    first.finish()
    old = read_release('dst')
    failed = RunRecorder.begin('dst', 'manual', config)
    store_product(failed)
    failed.finish(error=ValueError('model failed'))
    assert read_release('dst').release_id == old.release_id
    second = RunRecorder.begin('dst', 'manual', config)
    store_product(second)
    second.finish()
    assert read_release('dst').run_id == second.run_id
    assert read_release('dst', old.release_id) == old
    incomplete = RunRecorder.begin('solar-wind-speed', 'manual', config)
    store_product(incomplete, names=('plasma_speed_quantile',))
    with pytest.raises(ValueError, match='Incomplete'):
        incomplete.finish()
    incomplete.finish(error=ValueError('missing threshold artifact'))
    with pytest.raises(ReleaseNotFound):
        read_release('solar-wind-speed')


def test_publication_failure_does_not_roll_back_another_product(recorder_setup):
    dsn, passwords, config = recorder_setup
    success = RunRecorder.begin('solar-wind-density', 'manual', config)
    store_product(success, names=('plasma_density_quantile',))
    success.finish()
    failed = RunRecorder.begin('dst', 'manual', config)
    store_product(failed)
    with runtime(dsn, 'prophet', passwords) as conn:
        conn.execute("UPDATE prophet.forecast_artifact SET sha256=%s WHERE name='dst_quantile'", ('0' * 64,))
    with pytest.raises(ValueError, match='checksum'):
        failed.finish()
    assert read_release('solar-wind-density').run_id == success.run_id
    with pytest.raises(ReleaseNotFound):
        read_release('dst')
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (failed.run_id,)).fetchone()[0] == 'running'


def test_status_reports_product_failure_not_unrelated_success(recorder_setup):
    from argus_prophet.services.releases.status import product_status
    _, _, config = recorder_setup
    run = RunRecorder.begin('dst', 'manual', config)
    now = datetime(2026, 9, 13, 1, tzinfo=UTC)
    run.snapshot(ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[])))
    store_product(run)
    run.finish()
    failure = RunRecorder.begin('dst', 'manual', config)
    failure.finish(error=ValueError('model unavailable'))
    unrelated = RunRecorder.begin('solar-wind-density', 'manual', config)
    store_product(unrelated, names=('plasma_density_quantile',))
    unrelated.finish()
    status = product_status('dst', now=now)
    assert status.current_release.run_id == run.run_id
    assert status.current_release.input_diagnostics['normalized']['count'] == 0
    assert status.release_age_hours == 1 and status.freshness == 'unconfigured'
    assert status.latest_attempt.run_id == failure.run_id
    assert status.latest_attempt.status == 'failed'
    assert status.latest_attempt_artifacts == []
    assert product_status('solar-wind-density', now=now).latest_attempt.run_id == unrelated.run_id


def test_export_migration_preserves_release_bytes_on_upgrade_and_downgrade(recorder_setup):
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine
    from argus_prophet.db.session import get_database_url
    _, _, config = recorder_setup
    run = RunRecorder.begin('dst', 'manual', config)
    expected = store_product(run)
    run.finish()
    release = read_release('dst')
    path = Path(__file__).resolve().parents[2] / 'apps/prophet/src/argus_prophet/migrations/versions/20260918_remove_exports.py'
    spec = importlib.util.spec_from_file_location('remove_exports', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(get_database_url(), connect_args={'options': '-csearch_path=prophet,pg_catalog,pg_temp'})
    try:
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                migration.downgrade()
                assert conn.exec_driver_sql('SELECT count(*) FROM forecast_export').scalar() == 1
                migration.upgrade()
    finally:
        engine.dispose()
    assert read_release('dst') == release
    assert read_release('dst').artifacts[0].csv_text == expected
