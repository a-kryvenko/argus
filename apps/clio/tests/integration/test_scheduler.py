"""Clio scheduling against disposable PostgreSQL."""
from datetime import UTC, datetime, timedelta

import pytest
from domain_storage import database, migrate, runtime


def test_scheduler_retries_restarts_and_serializes_jobs(database, monkeypatch):
    dsns, urls, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    from clio.scheduling.jobs import execute, JobBusy
    from clio.db.locks import JOB_LOCKS
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    calls = []
    def fail():
        raise RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError, match='provider unavailable'):
        execute('refresh', fail, scheduled=True, now=now)
    assert execute('refresh', lambda: calls.append(1), scheduled=True, now=now)
    assert not execute('refresh', lambda: calls.append(2), scheduled=True, now=now)
    assert calls == [1]
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('SELECT pg_advisory_lock(%s)', (JOB_LOCKS['refresh'],))
        with pytest.raises(JobBusy):
            execute('refresh', lambda: calls.append(3), now=now)
    assert execute('refresh', lambda: calls.append(4), scheduled=True, now=now + timedelta(hours=1))
    assert calls == [1, 4]
