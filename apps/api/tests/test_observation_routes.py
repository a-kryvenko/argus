from datetime import UTC, datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from common.schemas.observation import Observation, ObservationPoint

from app.routers.public import observations


def test_history_returns_points_and_passes_limit(monkeypatch):
    point = ObservationPoint(
        issue_time=datetime(2026, 9, 5, tzinfo=UTC),
        bx=1, by=2, bz=-3, v=420, n=5, t=100000,
        kp=2, dst=-10, ap=5, f10_7=120,
    )
    loader = AsyncMock(return_value=Observation(points=[point]))
    monkeypatch.setattr(observations, "load_normalized_observations", loader)
    app = FastAPI()
    app.include_router(observations.router)
    session = object()
    app.dependency_overrides[observations.get_db_session] = lambda: session
    with TestClient(app) as client:
        response = client.get('/public/observations/history?limit=12')
        assert response.status_code == 200
        assert response.json()['data']['points'][0]['v'] == 420
        loader.assert_awaited_once_with(session, limit=12)
        for limit in [0, 169, 'invalid']:
            assert client.get(f'/public/observations/history?limit={limit}').status_code == 422
        loader.return_value = Observation(points=[])
        assert client.get('/public/observations/history').json() == {
            'success': True, 'data': {'points': []}, 'error': None,
        }
