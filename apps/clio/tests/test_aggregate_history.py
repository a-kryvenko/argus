import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from argus_clio.services.aggregate_history import history
from argus_clio.routers import solar_wind as routes

START = datetime(2026, 9, 1, tzinfo=UTC)


def test_saved_statistics_and_missing_buckets():
    stats = dict(count=3, invalid_count=1, missing_count=1, mean=-2, min=-8, max=3,
                 expected_count=5, coverage_percent=60, negative_count=2)
    row = SimpleNamespace(kind='mag', bucket_start=START, calculated_at=START,
        statistics={'window_complete': True, 'metrics': {'bz': stats},
                    'source_sequence': [{'spacecraft': 'A'}], 'source_changes': 0})
    session = AsyncMock()
    session.execute.side_effect = [Mock(scalars=Mock(return_value=[row])), Mock(scalars=Mock(return_value=[]))]
    data = asyncio.run(history(session, ['bz'], START, START+timedelta(minutes=10), 300))
    series = data['series']['bz']
    assert series['points'][0]['min'] == -8
    assert series['coverage']['percent'] == 60
    assert series['processing']['unavailable_buckets'] == 1
    assert series['processing']['available_buckets'] == 1
    assert series['coverage']['expected_slots'] == 5
    assert series['coverage']['missing_slots'] == 1
    assert series['coverage']['gaps'][0]['reason'] == 'partial'


def test_pending_recalculation_preserves_saved_values_and_coverage():
    row = SimpleNamespace(kind='mag', bucket_start=START, calculated_at=START,
        statistics={'window_complete': True, 'source_sequence': [{'spacecraft': 'A'}], 'source_changes': 0,
                    'metrics': {'bz': dict(count=5, invalid_count=0, missing_count=0, mean=-2, min=-8, max=3,
                                          expected_count=5, coverage_percent=100)}})
    session = AsyncMock()
    session.execute.side_effect = [Mock(scalars=Mock(return_value=[row])),
        Mock(scalars=Mock(return_value=[SimpleNamespace(kind='mag', hour=START)]))]
    data = asyncio.run(history(session, ['bz'], START, START+timedelta(minutes=5), 300))
    series = data['series']['bz']
    assert series['points'][0]['value'] == -2
    assert series['points'][0]['recalculation_pending']
    assert series['coverage']['percent'] == 100
    assert series['coverage']['gaps'] == []
    assert series['processing']['recalculation_pending_buckets'] == 1
    assert series['processing']['unavailable_buckets'] == 0


def test_partial_current_bucket_excluded_and_left_boundary_explicit():
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=Mock(return_value=[]))
    data = asyncio.run(history(session, ['bz'], START+timedelta(minutes=2), START+timedelta(minutes=8), 300, now=START+timedelta(minutes=7)))
    assert data['evaluated_from'] == START
    assert data['evaluated_to'] == START+timedelta(minutes=5)
    assert data['series']['bz']['coverage']['expected_slots'] == 0
    assert data['series']['bz']['coverage']['percent'] is None
    assert data['series']['bz']['coverage']['gaps'] == []
    assert data['series']['bz']['processing']['expected_buckets'] == 1


def test_resolution_routing_and_bounds(monkeypatch):
    app = FastAPI(); app.include_router(routes.router)
    app.dependency_overrides[routes.get_db_session] = lambda: object()
    aggregate = AsyncMock(return_value={})
    monkeypatch.setattr(routes.aggregate_history, 'history', aggregate)
    with TestClient(app) as client:
        base = '/internal/v1/observations/solar-wind/history?from=2026-08-01T00:00:00Z&to=2026-08-31T00:00:00Z'
        assert client.get(base).status_code == 422
        assert client.get(base+'&resolution=auto').status_code == 200
        assert aggregate.call_args.args[-1] == 3600
        assert client.get(base+'&resolution=5m').status_code == 200
        assert aggregate.call_args.args[-1] == 300
        assert client.get(base+'&resolution=bad').status_code == 422
