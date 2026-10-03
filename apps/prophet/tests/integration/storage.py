"""Fixtures and release builders for Prophet storage tests."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from domain_storage import database, migrate


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
    from argus_prophet.scheduling.jobs import generation_lock
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


