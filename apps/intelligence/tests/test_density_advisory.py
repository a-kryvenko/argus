"""Public infrastructure tests use canned backend responses, not physical assertions."""
import pytest
from fastapi.testclient import TestClient
from argus_intelligence import density_monitor
from argus_intelligence.main import app


def test_service_auth_and_sanitized_storage_failure(monkeypatch):
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN','test-token')
    monkeypatch.setattr(density_monitor,'list_profiles',lambda owner: [])
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
    assert density_monitor.retrieve_inputs()=={'atmospheric-density':None,'solar-wind-events':sentinel}
