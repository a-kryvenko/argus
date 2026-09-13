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
def recorder_setup(database, monkeypatch, tmp_path):
    dsn, passwords, environment = database
    with psycopg.connect(dsn) as conn:
        bootstrap.provision(conn, passwords)
    migrate(environment)
    for key in ('DB_NAME', 'DB_HOST', 'DB_PORT', 'PROPHET_DB_PASSWORD'):
        monkeypatch.setenv(key, environment[key])
    return dsn, passwords, SimpleNamespace(workdir=tmp_path, models_registry={})


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
    first = RunRecorder.begin('all', 'scheduled', config)
    # Caller has reacquired the existing generation lock after a writer crash.
    second = RunRecorder.begin('all', 'scheduled', config)
    second.finish(error=ValueError('bad model'))
    with runtime(dsn, 'prophet', passwords) as conn:
        rows = dict(conn.execute('SELECT id,status FROM prophet.forecast_run').fetchall())
        assert rows[first.run_id] == 'interrupted'
        assert rows[second.run_id] == 'failed'
