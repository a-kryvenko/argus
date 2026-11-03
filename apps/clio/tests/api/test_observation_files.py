import pytest
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.routers import observation_files as routes
from clio.routers.auth import require_service_token
from clio.observations import file_catalog

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def test_catalog_reads_causal_metadata_without_features_or_blobs(storage, monkeypatch):
    monkeypatch.setattr(file_catalog, 'aia_files', lambda *args, **kwargs: [])
    session = AsyncMock()
    from clio.observations import file_catalog as reading
    row = SimpleNamespace(slot_at=NOW, observed_at=NOW, available_at=NOW, sha256='a'*64,
        source_product='gong.live', file_id=__import__('uuid').uuid4())
    monkeypatch.setattr(reading, 'snapshot_receipt', lambda row: vars(row))
    monkeypatch.setattr(reading.file_observations, 'references', lambda ids: {row.file_id: dict(id=row.file_id, timestamp=NOW, bucket='argus-private', key='original.fits')})
    session.scalars.side_effect = [Mock(all=lambda: [row]), Mock(all=lambda: [])]
    files = asyncio.run(file_catalog.load_observation_files(session, NOW))
    assert len(files) == 1 and files[0].kind == 'gong'
    assert 'features' not in files[0].model_dump()
    for call in session.scalars.call_args_list:
        query = call.args[0].compile(dialect=postgresql.dialect())
        assert query.params['received_at_1'] == NOW
        assert 'fits_gzip' not in str(query) and 'gong_snapshot' not in str(query)



def client(monkeypatch, row):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    session = AsyncMock()
    session.scalars.return_value = Mock(first=lambda: row)
    app = FastAPI()
    app.include_router(routes.router, dependencies=[Depends(require_service_token)])
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    return TestClient(app), session


@pytest.mark.parametrize('kind', ['gong', 'goes', 'aia'])
def test_file_bytes_are_not_exposed(monkeypatch, kind):
    http, session = client(monkeypatch, None)
    response = http.get('/internal/v1/files/' + kind + '/' + 'a'*64,
                        headers={'Authorization': 'Bearer secret'})
    assert response.status_code == 404
    session.scalars.assert_not_called()


def test_catalog_http_uses_requested_kind_interval_and_cutoff(monkeypatch):
    http, session = client(monkeypatch, None)
    session.scalars.side_effect = lambda query: Mock(all=lambda: [])
    start = NOW-timedelta(days=7)
    result = http.get('/internal/v1/files', params={
        'kind': 'goes', 'start': start.isoformat(), 'end': NOW.isoformat(), 'as_of': NOW.isoformat()},
        headers={'Authorization': 'Bearer secret'})
    assert result.status_code == 200 and result.json()['data'] == []
    session.scalars.assert_awaited_once()
    query = session.scalars.call_args_list[0].args[0].compile(dialect=postgresql.dialect())
    assert 'file_observation' in str(query)
    assert query.params['kind_1'] == 'goes'
    assert query.params['source_key_1'] == start.replace(hour=0).isoformat()
    assert query.params['received_at_1'] == NOW
    assert http.get('/internal/v1/files', params={
        'kind': 'goes', 'start': NOW.isoformat(), 'end': start.isoformat(), 'as_of': NOW.isoformat()},
        headers={'Authorization': 'Bearer secret'}).status_code == 422
