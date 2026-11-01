from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clio.routers import collection_status as routes


def test_status_reads_observations_without_fetching_or_writing(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    session = AsyncMock()
    session.scalars.return_value = []
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    fetch = Mock(side_effect=AssertionError('Diagnostics must not fetch'))
    monkeypatch.setattr('requests.sessions.Session.request', fetch)
    with TestClient(app) as client:
        response = client.get('/internal/v1/observations/status')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    data = response.json()['data']
    assert set(data) == {'status', 'generated_at', 'measurements'}
    assert data['status'] == 'degraded'
    assert data['measurements'] and all(item['status'] == 'unavailable' for item in data['measurements'])
    session.scalars.assert_awaited_once()
    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()
    fetch.assert_not_called()
