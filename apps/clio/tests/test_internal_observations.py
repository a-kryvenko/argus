from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from argus_clio.routers import forecast_inputs as routes
from common.schemas.observation import Observation

NOW = datetime(2026, 9, 12, 0, tzinfo=UTC)


def client_with_session(monkeypatch):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    session = AsyncMock()
    session.execute.return_value = Mock(all=lambda: [SimpleNamespace(
        metric='dst', value=-10, observed_at=NOW, issue_time=NOW, v=420.)])
    loader = AsyncMock(return_value=Observation(points=[]))
    monkeypatch.setattr(routes, 'load_normalized_observations', loader)
    monkeypatch.setattr(routes, 'load_aia_features', AsyncMock(return_value=[]))
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_db_session] = lambda: session
    return TestClient(app), session, loader


def test_read_requires_service_credentials(monkeypatch):
    client, session, _ = client_with_session(monkeypatch)
    for headers in ({}, {'Authorization': 'Bearer wrong'}):
        assert client.get('/internal/v1/observations/forecast-inputs',
                          params={'as_of': NOW.isoformat()}, headers=headers).status_code == 401
    session.execute.assert_not_called()
    monkeypatch.delenv('OBSERVATIONS_SERVICE_TOKEN')
    assert client.get('/internal/v1/observations/forecast-inputs',
                      params={'as_of': NOW.isoformat()}).status_code == 503


def test_reads_both_sets_in_bounded_read_only_transaction(monkeypatch):
    client, session, loader = client_with_session(monkeypatch)
    response = client.get('/internal/v1/observations/forecast-inputs', params={'as_of': NOW.isoformat()},
                          headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['schema_version'] == 1
    assert payload['aia_frames'] == []
    routes.load_aia_features.assert_awaited_once_with(session, NOW)
    assert payload['measurements'][0]['metric'] == 'dst'
    assert payload['speed_observations'][0]['v'] == 420.
    statements = session.execute.call_args_list
    assert str(statements[0].args[0]) == 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'
    query = statements[2].args[0].compile(dialect=postgresql.dialect())
    assert set(query.params['metric_1']) == set(routes.DENSITY_METRICS)
    assert query.params['observed_at_2'] == NOW
    assert loader.call_args.kwargs['until'] == NOW
    assert (NOW - loader.call_args.kwargs['since']).days == 60
    speed_query = statements[3].args[0].compile(dialect=postgresql.dialect())
    assert speed_query.params['metric_1'] == 'v'
    assert speed_query.params['observed_at_2'] == NOW
    assert (NOW - speed_query.params['observed_at_1']).days == 60
    assert 'avg(clio.measurement.value)' in str(speed_query)
    assert 'GROUP BY date_trunc' in str(speed_query)
    session.rollback.assert_awaited_once()
    session.commit.assert_not_called()


def test_rejects_naive_and_future_cutoffs(monkeypatch):
    client, session, _ = client_with_session(monkeypatch)
    for cutoff in ('2026-09-12T00:00:00', '2999-01-01T00:00:00Z'):
        assert client.get('/internal/v1/observations/forecast-inputs', params={'as_of': cutoff},
                          headers={'Authorization': 'Bearer test-token'}).status_code == 422
    session.execute.assert_not_called()


def test_failed_read_releases_transaction(monkeypatch):
    import pytest
    client, session, loader = client_with_session(monkeypatch)
    loader.side_effect = RuntimeError('read failed')
    with pytest.raises(RuntimeError, match='read failed'):
        client.get('/internal/v1/observations/forecast-inputs', params={'as_of': NOW.isoformat()},
                   headers={'Authorization': 'Bearer test-token'})
    session.rollback.assert_awaited_once()
