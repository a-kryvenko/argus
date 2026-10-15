from types import SimpleNamespace
from uuid import uuid4
import httpx
import pytest
from fastapi import FastAPI,HTTPException
from fastapi.testclient import TestClient
from app.dashboard_auth import current_user
from app.db.session import get_db_session
from app.routers import density_advisory
from app.services.drag_monitor_client import request


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
