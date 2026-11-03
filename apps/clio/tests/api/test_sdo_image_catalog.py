from datetime import UTC, datetime, timedelta
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from clio.routers import observation_files as routes
from clio.routers.auth import require_service_token
from clio.observations import file_observations
from clio.ingestion.sdo_images import collect_one
SLOT = datetime(2026, 11, 2, 10, tzinfo=UTC)
NOW = SLOT+timedelta(minutes=30)


@pytest.fixture
def catalog(storage, monkeypatch, scientific_source):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    collect_one(SLOT, 'aia171', clock=lambda: NOW)
    app = FastAPI()
    app.include_router(routes.router, dependencies=[Depends(require_service_token)])
    app.dependency_overrides[routes.get_request_db_session] = lambda: None
    with TestClient(app) as client:
        yield client


def request(client, **params):
    return client.get('/internal/v1/files', headers={'Authorization':'Bearer secret'},
        params=dict(kind='aia', start=SLOT.isoformat(), end=(SLOT+timedelta(hours=1)).isoformat(), as_of=NOW.isoformat(), **{'channel':'aia171', **params}))


def test_catalog_returns_references_and_receipts(catalog):
    data = request(catalog).json()['data']
    assert len(data) == 1
    assert data[0]['file']['bucket'] == 'argus-private'
    assert data[0]['file']['key'].endswith('/original.fits')
    assert 'path' not in data[0]


def test_catalog_auth_and_channel_validation(catalog):
    assert catalog.get('/internal/v1/files').status_code == 401
    assert request(catalog, channel='other').status_code == 422


def test_invalid_db_metadata_is_service_failure(catalog, storage):
    file_observations.set_details('sdo', 'aia171', SLOT, {'invalid':'receipt'})
    assert request(catalog).status_code == 503


def test_time_filters_are_causal(catalog):
    response = catalog.get('/internal/v1/files', headers={'Authorization':'Bearer secret'},
        params=dict(kind='aia', channel='aia171', start=SLOT.isoformat(), end=(SLOT+timedelta(hours=1)).isoformat(), as_of=SLOT.isoformat()))
    assert response.status_code == 200
    assert not response.json()['data']
