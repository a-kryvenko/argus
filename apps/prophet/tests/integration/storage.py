"""Fixtures and release builders for Prophet storage tests."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from tests.support.database import database, migrate


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
    from prophet.scheduling.jobs import generation_lock
    with generation_lock() as writer:
        yield (*recorder_database, writer)


def store_product(run, names=('dst_quantile',), issue='2026-09-13T00:00:00+00:00'):
    from common.schemas.forecast_release import PREDICTION_COLUMNS
    valid = (datetime.fromisoformat(issue) + timedelta(hours=1)).isoformat()
    for name in names:
        if name == 'solar_wind_events':
            from pathlib import Path
            from forecast.inputs import ForecastInputs
            from common.schemas.observation import Observation
            from common.schemas.solar_wind_events import EventInputs
            from prophet.services.generation.calculation import CalculationRequest, calculate_product
            time = datetime.fromisoformat(issue)
            inputs = ForecastInputs(as_of=time, read_at=time, observations=Observation(points=[]),
                                    solar_wind_events=EventInputs(as_of=time, read_at=time))
            artifact = calculate_product(CalculationRequest('solar-wind-events', inputs, Path('/unused'), {}))[0]
            run.store(name, artifact.content, artifact.model_info, artifact.row_count, artifact.columns)
            content = artifact.content.decode()
            continue
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[name]]
        values = [issue, valid, '1', *('0' for _ in PREDICTION_COLUMNS[name])]
        content = ','.join(columns) + '\n' + ','.join(values) + '\n'
        run.store(name, content.encode('utf-8'), {}, 1, columns)
    return content


