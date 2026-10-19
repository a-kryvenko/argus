from datetime import UTC, datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from common.schemas.observation import Observation, ObservationPoint

from clio.routers import observations


def test_history_returns_points_and_passes_limit(monkeypatch):
    point = ObservationPoint(
        issue_time=datetime(2026, 9, 5, tzinfo=UTC),
        bx=1, by=2, bz=-3, v=420, n=5, t=100000,
        kp=2, dst=-10, ap=5, f10_7=120,
        s10=110.5, m10=99.25,
    )
    loader = AsyncMock(return_value=Observation(points=[point]))
    monkeypatch.setattr(observations, "load_normalized_observations", loader)
    app = FastAPI()
    app.include_router(observations.router)
    session = object()
    app.dependency_overrides[observations.get_db_session] = lambda: session
    with TestClient(app) as client:
        response = client.get('/internal/v1/observations/history?limit=12')
        assert response.status_code == 200
        assert response.json()['data']['points'][0]['v'] == 420
        assert response.json()['data']['points'][0]['s10'] == 110.5
        assert response.json()['data']['points'][0]['m10'] == 99.25
        assert response.json()['data']['points'][0]['y10'] is None
        loader.assert_awaited_once_with(session, limit=12)
        latest = client.get('/internal/v1/observations/latest')
        assert latest.status_code == 200
        assert latest.json()['data']['s10'] == 110.5
        assert latest.json()['data']['y10'] is None
        for limit in [0, 169, 'invalid']:
            assert client.get(f'/internal/v1/observations/history?limit={limit}').status_code == 422
        loader.return_value = Observation(points=[])
        assert client.get('/internal/v1/observations/history').json() == {
            'success': True, 'data': {'points': []}, 'error': None,
        }


def test_history_applies_exclusive_utc_bounds(monkeypatch):
    from datetime import timedelta
    loader = AsyncMock(return_value=Observation(points=[]))
    monkeypatch.setattr(observations, 'load_normalized_observations', loader)
    app = FastAPI()
    app.include_router(observations.router)
    session = object()
    app.dependency_overrides[observations.get_db_session] = lambda: session
    with TestClient(app) as client:
        result = client.get('/internal/v1/observations/history', params={
            'from': '2026-09-01T00:00:00Z', 'to': '2026-09-02T00:00:00Z'})
        assert result.status_code == 200
        loader.assert_awaited_once_with(session, limit=24, since=datetime(2026, 9, 1, tzinfo=UTC),
            until=datetime(2026, 9, 2, tzinfo=UTC) - timedelta(microseconds=1))
        assert client.get('/internal/v1/observations/history?from=2026-09-01').status_code == 422


def test_current_indices_keep_independent_source_times(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from sqlalchemy.dialects import postgresql
    at = datetime(2026, 10, 19, 15, tzinfo=UTC)
    flux_at = datetime(2026, 10, 18, 20, tzinfo=UTC)
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=lambda: Mock(all=lambda: [
        SimpleNamespace(metric='ap', value=4.0, observed_at=at, interval_end=at, quality='unverified'),
        SimpleNamespace(metric='f10_7', value=112.5, observed_at=flux_at, interval_end=None, quality=None),
    ]))
    loader = AsyncMock()
    monkeypatch.setattr(observations, 'load_normalized_observations', loader)
    app = FastAPI()
    app.include_router(observations.router)
    app.dependency_overrides[observations.get_db_session] = lambda: session
    with TestClient(app) as client:
        result = client.get('/internal/v1/observations/current-indices')
        assert result.status_code == 200
        series = result.json()['data']['series']
        assert series['ap']['observed_at'] == '2026-10-19T15:00:00Z'
        assert series['f10_7'] == {'observed_at': '2026-10-18T20:00:00Z', 'value': 112.5, 'quality': None}
        assert all(series[metric] is None for metric in ('s10', 'm10', 'y10'))
        assert 'issue_time' not in str(series)
        loader.assert_not_awaited()
        query = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
        assert 'DISTINCT ON (clio.measurement.metric)' in str(query)
        assert 'ORDER BY clio.measurement.metric, clio.measurement.observed_at DESC' in str(query)
        assert 'clio.measurement.observed_at <=' in str(query)
        assert set(query.params['param_1']) == {('ap', 'swpc.kp'), ('f10_7', 'swpc.f107'), ('f10_7', 'gfz.f107')}


def test_current_indices_preserve_missing_values_and_validate_selection():
    from unittest.mock import Mock
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=lambda: Mock(all=lambda: []))
    app = FastAPI()
    app.include_router(observations.router)
    app.dependency_overrides[observations.get_db_session] = lambda: session
    with TestClient(app) as client:
        result = client.get('/internal/v1/observations/current-indices?metrics=f10_7,s10,f10_7')
        assert result.status_code == 200
        assert result.json()['data']['series'] == {'f10_7': None, 's10': None}
        session.execute.reset_mock()
        for metrics in ('', 'unknown', 'f10_7,unknown'):
            assert client.get('/internal/v1/observations/current-indices', params={'metrics': metrics}).status_code == 422
        session.execute.assert_not_awaited()
