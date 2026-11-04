from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.routers import observations as routes
from clio.main import app as application

NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


def record(metric, minute, value):
    return SimpleNamespace(metric=metric, observed_at=NOW+timedelta(minutes=minute), value=value)


def result(records):
    return Mock(all=Mock(return_value=records), scalars=Mock(return_value=Mock(all=Mock(return_value=records))))


def test_only_two_numeric_routes_and_file_routes_remain():
    paths = set(application.openapi()['paths'])
    assert {path for path in paths if path.startswith('/internal/v1/')} == {
        '/internal/v1/files',
        '/internal/v1/observations/latest',
        '/internal/v1/observations/history',
        '/internal/v1/observations/sdo',
        '/internal/v1/observations/solar-wind-events/inputs',
    }


def test_clio_contract_has_no_as_of_parameter_or_response_field():
    spec = application.openapi()
    for path, methods in spec['paths'].items():
        if path.startswith('/internal/v1/'):
            assert all(parameter['name'] != 'as_of' for parameter in methods['get'].get('parameters', []))
    assert 'as_of' not in spec['components']['schemas']['EventInputs']['properties']


def test_latest_keeps_independent_times_and_null_latest_values():
    session = AsyncMock()
    session.execute.return_value = result([record('v', 1, 400), record('bx', 2, -3), record('n', 2, None)])
    data = asyncio.run(routes._read_latest(session, ['v', 'bx', 'n']))
    assert data == {'points': [{'time': NOW+timedelta(minutes=1), 'v': 400.},
                               {'time': NOW+timedelta(minutes=2), 'bx': -3., 'n': None}]}
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'DISTINCT ON' in sql and 'received_at <=' not in sql


def test_hourly_uses_normalized_storage_and_preserves_published_mean_dates():
    session = AsyncMock()
    session.execute.side_effect = [result([(NOW, -3.)]), result([record('s10_81mean', 30, 110)])]
    data = asyncio.run(routes._read_history(session, ['bx', 's10_81mean'], NOW, NOW+timedelta(hours=1),
                                    '1h'))
    assert data == {'interval': '1h', 'points': [
        {'time': NOW, 'bx': -3., 's10_81mean': None},
        {'time': NOW+timedelta(minutes=30), 'bx': None, 's10_81mean': 110.}]}
    sql = [str(call.args[0].compile(dialect=postgresql.dialect())) for call in session.execute.call_args_list]
    assert 'normalized_observation' in sql[0]
    assert 'measurement' in sql[1] and 'received_at <=' not in sql[1]


def test_five_minute_windows_use_valid_unique_minutes_and_exclude_partial_window():
    session = AsyncMock()
    session.execute.return_value = result([record('v', 0, 100), record('v', 1, 300),
        record('v', 2, float('nan')), record('v', 3, None), record('v', 5, 500)])
    data = asyncio.run(routes._read_history(session, ['v'], NOW, NOW+timedelta(minutes=7), '5m'))
    assert data == {'interval': '5m', 'points': [{'time': NOW, 'v': 200.}]}
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'source_product IN' in sql and 'normalized_observation' not in sql


def test_one_minute_does_not_fill_gaps_or_return_nonfinite_values():
    session = AsyncMock()
    session.execute.return_value = result([record('v', 0, 100), record('v', 2, float('inf'))])
    data = asyncio.run(routes._read_history(session, ['v'], NOW, NOW+timedelta(hours=1), '1m'))
    assert data['points'] == [{'time': NOW, 'v': 100.}, {'time': NOW+timedelta(minutes=2), 'v': None}]


@pytest.mark.parametrize('components, expected', [((-3., 4., 0.), 5.), ((-3., None, 0.), None)])
def test_hourly_total_field_requires_all_components(components, expected):
    session = AsyncMock()
    session.execute.return_value = result([(NOW, *components)])
    data = asyncio.run(routes._read_history(session, ['bt'], NOW, NOW+timedelta(hours=1),
                                           '1h'))
    assert data['points'] == [{'time': NOW, 'bt': expected}]


def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_request_db_session] = lambda: object()
    return TestClient(app)


def test_route_has_compact_payload_and_passes_selection(monkeypatch):
    read = AsyncMock(return_value={'interval': '5m', 'points': [{'time': NOW, 'v': 400.}]})
    monkeypatch.setattr(routes, '_read_history', read)
    response = client(monkeypatch).get('/internal/v1/observations/history', params={
        'metrics': 'v,v', 'from': NOW.isoformat(), 'to': (NOW+timedelta(hours=1)).isoformat(), 'interval': '5m'})
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json()['data'] == {'interval': '5m', 'points': [{'time': NOW.isoformat().replace('+00:00', 'Z'), 'v': 400.}]}
    assert read.call_args.args[1] == ['v']


@pytest.mark.parametrize('query', [
    {'metrics': 'unknown'}, {'interval': '10m'}, {'from': '2026-09-12'},
    {'to': '2026-09-11T12:00:00Z'}, {'from': '2025-01-01T00:00:00Z'},
])
def test_invalid_numeric_reads_fail_before_storage(monkeypatch, query):
    read = AsyncMock()
    monkeypatch.setattr(routes, '_read_history', read)
    response = client(monkeypatch).get('/internal/v1/observations/history', params={
        'metrics': 'v', 'from': NOW.isoformat(), 'to': (NOW+timedelta(hours=1)).isoformat(), **query})
    assert response.status_code == 422
    read.assert_not_awaited()


def test_hourly_indices_conform_to_integer_observation_contract():
    from clio.db.models.measurement import REQUIRED_METRICS
    from common.schemas.observation import Observation
    values = dict(bx=1., by=2., bz=-3., v=420., n=5., t=100000.,
                  kp=1.5533333333333332, dst=-10.75, ap=5.666666666666667, f10_7=98.39285714285714)
    session = AsyncMock()
    session.execute.return_value = result([(NOW, *(values[m] for m in REQUIRED_METRICS))])
    data = asyncio.run(routes._read_history(session, REQUIRED_METRICS, NOW, NOW+timedelta(hours=1),
                                    '1h'))
    point = data['points'][0]
    assert {metric: point[metric] for metric in routes.INTEGER_METRICS} == dict(kp=1, dst=-10, ap=5, f10_7=98)
    assert all(type(point[metric]) is int for metric in routes.INTEGER_METRICS)
    assert point['v'] == 420.
    Observation.model_validate({'points': [dict(issue_time=point['time'], **{m: point[m] for m in REQUIRED_METRICS})]})


def test_indices_are_converted_after_five_minute_aggregation():
    session = AsyncMock()
    session.execute.return_value = result([record('ap', 0, 1.9), record('ap', 1, 2.9),
                                          record('v', 0, 420.5), record('v', 1, 421.5)])
    data = asyncio.run(routes._read_history(session, ['ap', 'v'], NOW, NOW+timedelta(minutes=5),
                                    '5m'))
    assert data['points'] == [dict(time=NOW, ap=2, v=421.)]
    assert type(data['points'][0]['ap']) is int


def test_latest_indices_are_integers_and_missing_values_stay_null():
    session = AsyncMock()
    session.execute.return_value = result([record('kp', 0, 1.33), record('ap', 0, None),
                                          record('dst', 0, -10.75), record('f10_7', 0, 98.39)])
    data = asyncio.run(routes._read_latest(session, ['kp', 'ap', 'dst', 'f10_7']))
    assert data['points'] == [dict(time=NOW, kp=1, ap=None, dst=-10, f10_7=98)]
    assert type(data['points'][0]['kp']) is int


@pytest.mark.parametrize('endpoint,reader,params', [
    ('latest', '_read_latest', {}),
    ('history', '_read_history', {'from': NOW.isoformat(), 'to': (NOW+timedelta(hours=1)).isoformat()}),
])
def test_database_failure_returns_service_unavailable(monkeypatch, endpoint, reader, params):
    from sqlalchemy.exc import SQLAlchemyError
    monkeypatch.setattr(routes, reader, AsyncMock(side_effect=SQLAlchemyError('private database details')))
    response = client(monkeypatch).get('/internal/v1/observations/' + endpoint, params=params)
    assert response.status_code == 503
    assert 'private database details' not in response.text
