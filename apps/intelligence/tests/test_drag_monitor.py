import asyncio
from unittest.mock import AsyncMock
import pytest
from pydantic import ValidationError
from common.schemas.drag_monitor import DragProfileInput
from intelligence import monitor
from intelligence.main import app
from fastapi.testclient import TestClient


def profile_body():
    return DragProfileInput(name='Satellite A', altitude_km=400, inclination_deg=51.6, mass_kg=100,
        horizon_hours=24, thresholds={'elevated_altitude_loss_m': 50, 'high_altitude_loss_m': 100},
        modes=[{'name': 'Nominal', 'effective_area_m2': 1, 'drag_coefficient': 2.2}], active_mode='Nominal')


def test_mode_and_threshold_validation():
    body = profile_body().model_dump()
    for changes in ({'active_mode': 'Missing'}, {'modes': body['modes'] * 2},
                    {'thresholds': {'elevated_altitude_loss_m': 100, 'high_altitude_loss_m': 50}},
                    {'name': '  '}, {'mass_kg': 0}, {'modes': []}):
        with pytest.raises(ValidationError):
            DragProfileInput.model_validate({**body, **changes})


def test_monitor_routes_require_service_auth_and_sanitize_storage_errors(monkeypatch):
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(monitor, 'list_profiles', AsyncMock(return_value=[]))
    with TestClient(app) as client:
        path = '/internal/v1/drag-monitor/7'
        assert client.get(path).status_code == 401
        assert client.get(path, headers={'Authorization': 'Bearer test-token'}).json() == []
        def fail(owner):
            raise RuntimeError('secret database configuration')
        monkeypatch.setattr(monitor, 'list_profiles', fail)
        response = client.get(path, headers={'Authorization': 'Bearer test-token'})
        assert response.status_code == 503
        assert 'secret' not in response.text


