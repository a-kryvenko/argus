from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from common.schemas.leo_drag import LeoDragRequest
from app.routers import leo_drag
from app.services import intelligence_client as service
from app.services.forecast_errors import ArtifactNotReadyError

REQUEST = dict(altitude_km=400, inclination_deg=51.6, mass_kg=100,
               effective_area_m2=1, drag_coefficient=2.2, horizon_hours=24)


@pytest.fixture
def result(monkeypatch):
    monkeypatch.setenv('INTELLIGENCE_URL', 'http://intelligence-api:8000')
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'impact-token')
    now = datetime.now(UTC)
    point = dict(mean_density_kg_m3=1e-12, mean_drag_accel_m_s2=1e-6,
                 mean_along_track_deceleration_m_s2=1e-6, delta_v_loss_m_s=0.1,
                 estimated_altitude_loss_m=100)
    return dict(model='circular_leo_drag_v1', computed_at=now.isoformat(), start_time=now.isoformat(),
                end_time=(now+timedelta(hours=24)).isoformat(), inputs=dict(REQUEST),
                source=dict(release_id=str(uuid4()), issue_time=now.isoformat(), observed_at=now.isoformat(),
                            dtc_observed_at=now.isoformat(), history_start=(now-timedelta(days=81)).isoformat(),
                            background_method='trailing_81_daily_values', background_interpolated_days={},
                            dtc_method='causal_dst_ap_v1'),
                mean_density_kg_m3=1e-12, mean_drag_accel_m_s2=1e-6,
                delta_v_loss_m_s=0.1, estimated_altitude_loss_m=100,
                drag_risk='not_assessed', risk_reason='No thresholds', assumptions=['Circular orbit'],
                predictions=[dict(valid_time=(now+timedelta(hours=i)).isoformat(), lead_hours=i,
                                  **{**point, 'delta_v_loss_m_s': i/240, 'estimated_altitude_loss_m': 100*i/24}) for i in range(25)])


def client():
    app = FastAPI()
    app.include_router(leo_drag.router)
    return TestClient(app)


def test_http_contract_and_public_envelope(result, monkeypatch):
    def respond(request):
        assert request.url.path == '/internal/v1/risks/leo-drag'
        assert request.headers['Authorization'] == 'Bearer impact-token'
        assert request.method == 'POST'
        return httpx.Response(200, json=result)
    monkeypatch.setattr(leo_drag, 'assess_drag', lambda request: service.assess_drag(request, transport=httpx.MockTransport(respond)))
    with client() as api:
        response = api.post('/public/risks/leo-drag', json=REQUEST)
        assert response.status_code == 200
        assert response.json()['data']['estimated_altitude_loss_m'] == 100
        assert response.json()['success'] is True
        compact = response.json()['data']
        assert 'meta' not in compact and 'model' not in compact
        assert 'background_method' not in compact['source']
        assert compact['source']['release_id'] == result['source']['release_id']
        assert compact['source']['driver_mode'] == 'observed_persistence'
        assert compact['source']['background_interpolated'] is False
        expanded = api.post('/public/risks/leo-drag?meta=true', json=REQUEST).json()['data']
        metadata = expanded.pop('meta')
        assert metadata['source']['model'] == 'JB2008'
        assert metadata['source']['dtc_method'] == 'causal_dst_ap_v1'
        assert expanded == compact
        assert '/public/risks/leo-drag' in api.get('/openapi.json').json()['paths']


@pytest.mark.parametrize('change', [dict(mass_kg=0), dict(effective_area_m2=-1), dict(drag_coefficient=0),
                                  dict(altitude_km=199), dict(altitude_km=801), dict(inclination_deg=181),
                                  dict(horizon_hours=25), dict(eccentricity=0.1), dict(mass_kg='NaN'),
                                  dict(thresholds=dict(elevated_altitude_loss_m=100, high_altitude_loss_m=50))])
def test_invalid_inputs_do_not_call_service(change, monkeypatch):
    def unexpected(_):
        pytest.fail('Invalid input reached impact service')
    monkeypatch.setattr(leo_drag, 'assess_drag', unexpected)
    with client() as api:
        assert api.post('/public/risks/leo-drag', json={**REQUEST, **change}).status_code == 422


@pytest.mark.parametrize('status', [401, 404, 500, 503, 302])
def test_upstream_errors_are_unavailable(result, status):
    with pytest.raises(ArtifactNotReadyError):
        service.assess_drag(LeoDragRequest(**REQUEST), transport=httpx.MockTransport(lambda r: httpx.Response(status)))


@pytest.mark.parametrize('failure', ['malformed', 'mismatch', 'partial', 'risk', 'large', 'missing-config', 'timeout'])
def test_unusable_responses(result, failure, monkeypatch):
    if failure == 'malformed': result = {'bad': 'payload'}
    elif failure == 'mismatch': result['inputs']['mass_kg'] = 200
    elif failure == 'partial': result['predictions'].pop()
    elif failure == 'risk': result['drag_risk'] = 'high'
    elif failure == 'large': monkeypatch.setattr(service, 'MAX_RESPONSE_BYTES', 10)
    elif failure == 'missing-config': monkeypatch.delenv('INTELLIGENCE_SERVICE_TOKEN')
    def respond(request):
        if failure == 'timeout': raise httpx.ReadTimeout('secret upstream info')
        return httpx.Response(200, json=result)
    with pytest.raises(ArtifactNotReadyError):
        service.assess_drag(LeoDragRequest(**REQUEST), transport=httpx.MockTransport(respond))


@pytest.mark.parametrize(('error', 'status'), [(ArtifactNotReadyError('private detail'), 503), (service.DragDomainError('Outside grid'), 422)])
def test_public_error_mapping(error, status, monkeypatch):
    def fail(_): raise error
    monkeypatch.setattr(leo_drag, 'assess_drag', fail)
    with client() as api:
        response = api.post('/public/risks/leo-drag', json=REQUEST)
    assert response.status_code == status
    assert 'private detail' not in response.text
