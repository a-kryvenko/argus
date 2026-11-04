import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.routers import observation_files as routes
from clio.routers.auth import require_service_token

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


@pytest.mark.parametrize('observation', ['gong', 'goes', 'aia_094_nrt', 'hmi_m_nrt', 'aia_193_browse'])
def test_catalog_reads_common_metadata_in_one_query(observation):
    session = AsyncMock()
    row = dict(observed_at=NOW, details={'slot_at': NOW.isoformat(), 'sha256': 'a'*64},
        source_product='source', file_id=uuid4(), file_timestamp=NOW,
        bucket='argus-private', key='original')
    session.execute.return_value = Mock(mappings=lambda: Mock(all=lambda: [row]))
    files = asyncio.run(routes.load_observation_files(session, observation=observation,
        start=NOW-timedelta(days=1), end=NOW+timedelta(hours=1)))
    assert len(files) == 1 and files[0].observation == observation
    assert files[0].observed_at == NOW and files[0].slot_at == NOW
    assert set(files[0].model_dump()) == {'observation', 'slot_at', 'observed_at', 'sha256', 'source_product', 'file'}
    session.execute.assert_awaited_once()
    query = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert 'JOIN files' in str(query)
    assert 'received_at <=' not in str(query)


def client(monkeypatch, rows=()):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    session = AsyncMock()
    session.execute.return_value = Mock(mappings=lambda: Mock(all=lambda: list(rows)))
    app = FastAPI()
    app.include_router(routes.router, dependencies=[Depends(require_service_token)])
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    return TestClient(app), session


def request(http, **params):
    return http.get('/internal/v1/files', params={
        'observation': 'goes', 'start': (NOW-timedelta(days=7)).isoformat(), 'end': NOW.isoformat(), **params},
        headers={'Authorization': 'Bearer secret'})


def test_catalog_http_uses_observation_and_interval(monkeypatch):
    http, session = client(monkeypatch)
    response = request(http)
    assert response.status_code == 200 and response.json()['data'] == []
    assert response.headers['cache-control'] == 'no-store'
    session.execute.assert_awaited_once()
    query = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert query.params['kind_1'] == 'goes'
    assert query.params['entity_1'] == 'json.original'
    assert query.params['observed_at_1'] == NOW-timedelta(days=7)
    assert request(http, start=NOW.isoformat(), end=(NOW-timedelta(days=1)).isoformat()).status_code == 422


@pytest.mark.parametrize('observation', ['unknown', 'v', 'aia'])
def test_invalid_file_observation_fails_before_storage(monkeypatch, observation):
    http, session = client(monkeypatch)
    assert request(http, observation=observation).status_code == 422
    session.execute.assert_not_awaited()


def test_catalog_rejects_overflow_before_decoding(monkeypatch):
    http, session = client(monkeypatch, [object()] * (routes.MAX_FILES+1))
    assert request(http).status_code == 503


@pytest.mark.parametrize('column', ['observed_at', 'file_timestamp'])
def test_catalog_does_not_repair_timezone_naive_database_values(monkeypatch, column):
    row = dict(observed_at=NOW, details={'slot_at': NOW.isoformat(), 'sha256': 'a'*64},
        source_product='goes.live', file_id=uuid4(), file_timestamp=NOW,
        bucket='argus-private', key='original')
    row[column] = NOW.replace(tzinfo=None)
    http, _ = client(monkeypatch, [row])
    assert request(http).status_code == 503
