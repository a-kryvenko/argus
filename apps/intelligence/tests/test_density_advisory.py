import asyncio
from unittest.mock import AsyncMock
"""Public infrastructure tests use canned backend responses, not physical assertions."""
import pytest
from fastapi.testclient import TestClient
from intelligence import density_monitor
from intelligence.main import app


def test_delete_uses_authenticated_owner_revision_and_reports_conflicts(monkeypatch):
    from uuid import uuid4
    from intelligence.monitor import ProfileConflict, ProfileNotFound

    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'test-token')
    profile_id = uuid4()
    path = f'/internal/v1/density-advisory/7/{profile_id}'
    headers = {'Authorization': 'Bearer test-token'}
    calls = []
    async def remove(owner_id, target, *, revision):
        calls.append((owner_id, target, revision))
        return {'deleted': True}
    monkeypatch.setattr(density_monitor, 'delete_profile', remove)
    with TestClient(app) as client:
        assert client.delete(path+'?revision=1').status_code == 401
        assert client.delete(path, headers=headers).status_code == 422
        assert client.delete(path+'?revision=0', headers=headers).status_code == 422
        response = client.delete(path+'?revision=3', headers=headers)
        assert response.status_code == 200 and response.json() == {'deleted': True}
        assert calls == [(7, profile_id, 3)]
        for error, status in [(ProfileNotFound('Spacecraft not found'), 404), (ProfileConflict('Profile changed'), 409)]:
            def fail(*args, **kwargs):
                raise error
            monkeypatch.setattr(density_monitor, 'delete_profile', fail)
            assert client.delete(path+'?revision=3', headers=headers).status_code == status


def test_service_auth_and_sanitized_storage_failure(monkeypatch):
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN','test-token')
    monkeypatch.setattr(density_monitor,'list_profiles',AsyncMock(return_value=[]))
    with TestClient(app) as client:
        path='/internal/v1/density-advisory/7'
        assert client.get(path).status_code==401
        headers={'Authorization':'Bearer test-token'}
        assert client.get(path,headers=headers).json()==[]
        def fail(owner): raise RuntimeError('secret database connection')
        monkeypatch.setattr(density_monitor,'list_profiles',fail)
        response=client.get(path,headers=headers)
        assert response.status_code==503 and 'secret' not in response.text


def test_core_input_error_is_422(monkeypatch):
    from intelligence_core.api import AdvisoryInputError
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN','test-token')
    def fail(*args,**kwargs): raise AdvisoryInputError('Trajectory times must be unique and strictly increasing')
    monkeypatch.setattr(density_monitor,'assess_profile',fail)
    with TestClient(app) as client:
        response=client.post('/internal/v1/density-advisory/7/00000000-0000-0000-0000-000000000001/assess',headers={'Authorization':'Bearer test-token'})
        assert response.status_code==422


def test_input_retrieval_failure_does_not_discard_other_product(monkeypatch):
    sentinel=object()
    def fetch(product):
        if product=='atmospheric-density': raise RuntimeError('secret')
        return sentinel
    monkeypatch.setattr(density_monitor,'fetch_release',fetch)
    assert asyncio.run(density_monitor.retrieve_inputs())=={'atmospheric-density':None,'solar-wind-events':sentinel}
