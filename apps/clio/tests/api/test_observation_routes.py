from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.routers import observations as routes
from clio.observations import metrics as reads
from clio.main import app as application

NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


def record(metric, minute, value, quality=None):
    return SimpleNamespace(metric=metric, observed_at=NOW+timedelta(minutes=minute), value=value, quality=quality)


def result(records):
    return Mock(all=Mock(return_value=records), scalars=Mock(return_value=Mock(all=Mock(return_value=records))))


def test_only_two_numeric_routes_and_file_routes_remain():
    paths = set(application.openapi()['paths'])
    numeric = {path for path in paths if path.startswith('/internal/v1/observations')
               and not any(part in path for part in ('/files', '/sdo-', '/solar-wind-events/'))}
    assert numeric == {'/internal/v1/observations/latest', '/internal/v1/observations/metrics'}


def test_latest_keeps_independent_times_and_null_latest_values():
    session = AsyncMock()
    session.execute.return_value = result([record('v', 1, 400), record('bx', 2, -3), record('n', 2, 10, 'flagged')])
    data = asyncio.run(reads.latest(session, ['v', 'bx', 'n'], NOW+timedelta(minutes=3)))
    assert data == {'points': [{'time': NOW+timedelta(minutes=1), 'v': 400.},
                               {'time': NOW+timedelta(minutes=2), 'bx': -3., 'n': None}]}
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'DISTINCT ON' in sql and 'received_at <=' in sql


def test_hourly_uses_normalized_storage_and_preserves_published_mean_dates():
    session = AsyncMock()
    session.execute.side_effect = [result([(NOW, -3.)]), result([record('s10_81mean', 30, 110)])]
    data = asyncio.run(reads.history(session, ['bx', 's10_81mean'], NOW, NOW+timedelta(hours=1),
                                    '1h', NOW+timedelta(hours=2)))
    assert data == {'interval': '1h', 'points': [
        {'time': NOW, 'bx': -3., 's10_81mean': None},
        {'time': NOW+timedelta(minutes=30), 'bx': None, 's10_81mean': 110.}]}
    sql = [str(call.args[0].compile(dialect=postgresql.dialect())) for call in session.execute.call_args_list]
    assert 'normalized_observation' in sql[0]
    assert 'measurement' in sql[1] and 'received_at <=' in sql[1]


def test_five_minute_windows_use_valid_unique_minutes_and_exclude_partial_window():
    session = AsyncMock()
    session.execute.return_value = result([record('v', 0, 100), record('v', 1, 300),
        record('v', 2, float('nan')), record('v', 3, 900, 'flagged'), record('v', 5, 500)])
    data = asyncio.run(reads.history(session, ['v'], NOW, NOW+timedelta(minutes=7), '5m', NOW+timedelta(minutes=7)))
    assert data == {'interval': '5m', 'points': [{'time': NOW, 'v': 200.}]}
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'source_product IN' in sql and 'normalized_observation' not in sql


def test_one_minute_does_not_fill_gaps_or_return_nonfinite_values():
    session = AsyncMock()
    session.execute.return_value = result([record('v', 0, 100), record('v', 2, float('inf'))])
    data = asyncio.run(reads.history(session, ['v'], NOW, NOW+timedelta(hours=1), '1m', NOW+timedelta(hours=1)))
    assert data['points'] == [{'time': NOW, 'v': 100.}, {'time': NOW+timedelta(minutes=2), 'v': None}]


def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_request_db_session] = lambda: object()
    return TestClient(app)


def test_route_has_compact_payload_and_passes_selection(monkeypatch):
    read = AsyncMock(return_value={'interval': '5m', 'points': [{'time': NOW, 'v': 400.}]})
    monkeypatch.setattr(reads, 'history', read)
    response = client(monkeypatch).get('/internal/v1/observations/metrics', params={
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
    monkeypatch.setattr(reads, 'history', read)
    response = client(monkeypatch).get('/internal/v1/observations/metrics', params={
        'metrics': 'v', 'from': NOW.isoformat(), 'to': (NOW+timedelta(hours=1)).isoformat(), **query})
    assert response.status_code == 422
    read.assert_not_awaited()
