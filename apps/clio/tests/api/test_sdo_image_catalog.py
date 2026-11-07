from datetime import UTC, datetime, timedelta
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from clio.routers import observation_files as routes
from clio.routers.auth import require_service_token
from sqlalchemy import update
from clio.db.models.file_observation import FileObservation
from clio.ingestion.extras.sdo import collect_one
SLOT = datetime(2026, 11, 2, 10, tzinfo=UTC)
NOW = SLOT+timedelta(minutes=30)


@pytest.fixture
def catalog(storage, monkeypatch, scientific_source, metadata_session, request):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    image, metadata, original = scientific_source.download.return_value
    metadata['observed_at'] = (SLOT+timedelta(minutes=getattr(request, 'param', 0))).isoformat()
    collect_one(SLOT, 'aia171', clock=lambda: NOW)
    app = FastAPI()
    app.include_router(routes.router, dependencies=[Depends(require_service_token)])
    app.dependency_overrides[routes.get_request_db_session] = lambda: metadata_session
    with TestClient(app) as client:
        yield client


def request(client, **params):
    return client.get('/internal/v1/files', headers={'Authorization':'Bearer secret'},
        params={'observation':'aia_171_nrt', 'start':SLOT.isoformat(), 'end':(SLOT+timedelta(hours=1)).isoformat(),
                **params})


def test_catalog_returns_s3_references(catalog):
    data = request(catalog).json()['data']
    assert len(data) == 1
    assert data[0]['file']['bucket'] == 'argus-private'
    assert data[0]['file']['key'].endswith('/original.fits')
    assert data[0]['observation'] == 'aia_171_nrt'
    assert 'path' not in data[0]
    assert 'available_at' not in data[0]


def test_catalog_uses_request_session_without_sync_connection(catalog, monkeypatch, metadata_session):
    from clio.db import session as db
    from unittest.mock import Mock

    with monkeypatch.context() as patch:
        patch.setattr(db, 'get_sync_engine', Mock(side_effect=AssertionError('Use the request session')))
        assert request(catalog).status_code == 200
    metadata_session.execute.assert_awaited_once()


def test_catalog_auth_and_observation_validation(catalog):
    assert catalog.get('/internal/v1/files').status_code == 401
    assert request(catalog, observation='other').json()['data'] == []


def test_source_metadata_does_not_affect_api(catalog, storage):
    with storage.engine.begin() as connection:
        connection.execute(update(FileObservation).values(details={'source_metadata': 'irrelevant'}))
    assert len(request(catalog).json()['data']) == 1


def test_time_filters_use_observation_time(catalog):
    response = catalog.get('/internal/v1/files', headers={'Authorization':'Bearer secret'},
        params=dict(observation='aia_171_nrt', start=SLOT.isoformat(), end=(SLOT+timedelta(hours=1)).isoformat()))
    assert response.status_code == 200
    assert len(response.json()['data']) == 1


@pytest.mark.parametrize('catalog', [-1], indirect=True)
def test_observed_time_before_slot_is_not_lost_at_end_boundary(catalog):
    response = request(catalog, start=(SLOT-timedelta(minutes=2)).isoformat(),
                       end=(SLOT-timedelta(seconds=30)).isoformat())
    assert response.status_code == 200
    assert len(response.json()['data']) == 1
