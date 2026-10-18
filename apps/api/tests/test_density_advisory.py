from types import SimpleNamespace
from uuid import uuid4
import httpx
import pytest
from fastapi import FastAPI,HTTPException
from fastapi.testclient import TestClient
from api.services.authentication import current_user
from api.db.session import get_db_session
from api.routers import density_advisory
from api.services.intelligence.profiles import request


def test_session_ownership_origin_and_exact_source_routes(monkeypatch):
    app=FastAPI()
    app.include_router(density_advisory.router)
    async def no_database(): yield None
    app.dependency_overrides[get_db_session]=no_database
    calls=[]
    monkeypatch.setattr(density_advisory,'request',lambda *a,**k:calls.append((a,k)) or [])
    base='/dashboard/density-advisory'
    with TestClient(app) as client:
        assert client.get(base).status_code==401
        app.dependency_overrides[current_user]=lambda:SimpleNamespace(id=7)
        assert client.get(base+'?owner_id=999').status_code==200
        assert calls[-1][0][1]==7
        profile,run,event=uuid4(),uuid4(),uuid4()
        assert client.delete(base+f'/{profile}?revision=1').status_code==403
        assert client.delete(base+f'/{profile}',headers={'Origin':'http://localhost:3000'}).status_code==422
        assert client.delete(base+f'/{profile}?revision=1&owner_id=999',headers={'Origin':'http://localhost:3000'}).status_code==200
        assert calls[-1][0][:3]==('DELETE',7,f'/{profile}')
        assert calls[-1][1]['params']=={'revision':1}
        for suffix in (f'/{profile}/assess',f'/events/{event}/ack'):
            assert client.post(base+suffix).status_code==403
            assert client.post(base+suffix,headers={'Origin':'http://localhost:3000'}).status_code==200
            assert calls[-1][0][1:3]==(7,suffix)
        assert client.get(base+f'/{profile}/runs/{run}').status_code==200
        assert calls[-1][0][1:3]==(7,f'/{profile}/runs/{run}')
        suffix = f'/{profile}/runs/{run}/inputs/atmospheric-density'
        assert client.get(base+suffix+'?owner_id=999').status_code==200
        assert calls[-1][0][1:3]==(7,suffix)
        assert calls[-1][1]['max_response_bytes']==64*1024*1024
        assert client.get(base+f'/{profile}/runs/{run}/inputs/unsupported').status_code==422
        assert client.get(base+'/events?page=2').status_code==200
        assert calls[-1][1]['params']=={'page':2}


@pytest.mark.parametrize('status,expected',[(422,422),(404,404),(409,409),(500,503)])
def test_density_transport_paths_and_sanitized_errors(monkeypatch,status,expected):
    monkeypatch.setenv('INTELLIGENCE_URL','http://internal')
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN','test-token')
    def respond(req):
        assert req.url.path=='/internal/v1/density-advisory/7'
        assert req.headers['Authorization']=='Bearer test-token'
        return httpx.Response(status,json={'detail':'secret upstream error'})
    with pytest.raises(HTTPException) as raised:
        request('GET',7,'',list,resource='density-advisory',transport=httpx.MockTransport(respond))
    assert raised.value.status_code==expected
    assert 'secret' not in raised.value.detail


@pytest.mark.parametrize('detail, expected', [
    ('High density ratio must exceed review density ratio', 'High density ratio must exceed review density ratio.'),
    ('Apoapsis, periapsis and inclination are required', 'Apoapsis, periapsis and inclination are required.'),
    ('secret upstream error', 'Invalid spacecraft parameters.'),
    ([{'loc': ['body', 'trajectory'], 'msg': 'Field required', 'input': 'secret'}], 'Invalid spacecraft parameters.'),
])
def test_density_validation_errors_describe_spacecraft_and_hide_upstream_inputs(monkeypatch, detail, expected):
    monkeypatch.setenv('INTELLIGENCE_URL', 'http://internal')
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'test-token')
    transport = httpx.MockTransport(lambda req: httpx.Response(422, json={'detail': detail}))
    with pytest.raises(HTTPException) as raised:
        request('POST', 7, '', dict, resource='density-advisory', transport=transport)
    assert raised.value.status_code == 422
    assert raised.value.detail.startswith(expected)
    assert 'secret' not in raised.value.detail and 'trajectory' not in raised.value.detail


@pytest.mark.parametrize('method', ['POST', 'PUT'])
def test_spacecraft_writes_need_orbit_but_no_trajectory(monkeypatch, method):
    app = FastAPI()
    app.include_router(density_advisory.router)
    app.dependency_overrides[current_user] = lambda: SimpleNamespace(id=7)
    calls = []
    monkeypatch.setattr(density_advisory, 'request', lambda *a, **k: calls.append((a, k)) or {})
    url = '/dashboard/density-advisory'
    if method == 'PUT':
        url += f'/{uuid4()}?revision=1'
    body = dict(name='Satellite A', apoapsis_km=420, periapsis_km=400, inclination_deg=51.6)
    headers = {'Origin': 'http://localhost:3000'}
    with TestClient(app) as client:
        incomplete = {key: value for key, value in body.items() if key != 'inclination_deg'}
        assert client.request(method, url, json=incomplete, headers=headers).status_code == 422
        assert not calls
        response = client.request(method, url, json=body, headers=headers)
        assert response.status_code == (201 if method == 'POST' else 200)
        forwarded = calls[-1][1]['body']
        assert forwarded.trajectory is None and forwarded.reference is None
        assert forwarded.reference_mode == 'prophet_background'
        assert 'max_sample_gap_minutes' not in forwarded.policy.model_dump()
