from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.routers import measurements as routes

NOW = datetime(2026, 9, 12, tzinfo=UTC)


def client_with_session(monkeypatch):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    session = AsyncMock()
    session.execute.return_value = Mock(all=lambda: [])
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    return TestClient(app), session


def params(**kwargs):
    return {'start': (NOW-timedelta(days=3)).isoformat(), 'end': NOW.isoformat(),
            'metrics': 'bx,t', **kwargs}


def test_read_requires_service_credentials(monkeypatch):
    client, session = client_with_session(monkeypatch)
    for headers in ({}, {'Authorization': 'Bearer wrong'}):
        assert client.get('/internal/v1/observations/measurements', params=params(), headers=headers).status_code == 401
    session.execute.assert_not_called()
    monkeypatch.delenv('OBSERVATIONS_SERVICE_TOKEN')
    assert client.get('/internal/v1/observations/measurements', params=params()).status_code == 503


def test_measurement_selection_and_aggregation_are_requested_by_caller(monkeypatch):
    client, session = client_with_session(monkeypatch)
    response = client.get('/internal/v1/observations/measurements',
        params=params(aggregation='hourly', min_value=0, max_value=1000, exclude_value=999),
        headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 200, response.text
    assert response.json()['data'] == {'schema_version': 1, 'items': []}
    query = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert query.params['metric_1'] == ['bx', 't']
    assert query.params['observed_at_1'] == NOW-timedelta(days=3)
    assert query.params['observed_at_2'] == NOW
    assert 'avg(measurement.value)' in str(query)
    assert 'GROUP BY' in str(query)
    assert 'measurement.observed_at < ' in str(query)
    assert {0, 1000, 999} <= set(v for v in query.params.values() if isinstance(v, (float, int)))


def test_raw_measurements_preserve_values_and_availability(monkeypatch):
    client, session = client_with_session(monkeypatch)
    session.execute.return_value = Mock(all=lambda: [Mock(_mapping={
        'metric': 'bx', 'value': -3., 'observed_at': NOW, 'available_at': NOW})])
    response = client.get('/internal/v1/observations/measurements', params=params(),
                          headers={'Authorization': 'Bearer test-token'})
    assert response.json()['data']['items'][0]['value'] == -3.
    sql = str(session.execute.call_args.args[0])
    assert 'avg(' not in sql and 'greatest(' in sql


def test_rejects_naive_future_and_invalid_intervals(monkeypatch):
    client, session = client_with_session(monkeypatch)
    for changes in ({'start': '2026-09-12T00:00:00'}, {'end': '2999-01-01T00:00:00Z'},
                    {'start': NOW.isoformat()}, {'metrics': ','}, {'min_value': 2, 'max_value': 1}):
        assert client.get('/internal/v1/observations/measurements', params=params(**changes),
                          headers={'Authorization': 'Bearer test-token'}).status_code == 422
    session.execute.assert_not_called()


def test_oversized_reads_fail_without_truncation(monkeypatch):
    client, session = client_with_session(monkeypatch)
    session.execute.return_value = Mock(all=lambda: [None]*250_001)
    assert client.get('/internal/v1/observations/measurements', params=params(),
                      headers={'Authorization': 'Bearer test-token'}).status_code == 503
