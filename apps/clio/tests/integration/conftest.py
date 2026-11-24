import shutil

import httpx
import pytest
from common import config
from tests.support.clio import clio_server
from tests.support.database import ROOT, database


@pytest.fixture
def project_config(tmp_path, monkeypatch):
    directory = tmp_path / 'configs'
    directory.mkdir(exist_ok=True)
    for name in ('project.yaml', 'models_registry.yaml'):
        shutil.copyfile(ROOT / 'configs' / name, directory / name)
    monkeypatch.setenv('ARGUS_WORKDIR', str(tmp_path))
    monkeypatch.setattr(config, '_config', None)


@pytest.fixture
def file_database(database, project_config, monkeypatch, object_store):
    _, _, environment = database
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db import session as db
    from clio.db.models.file import File
    from clio.db.models.file_observation import FileObservation
    from clio.db.models.measurement import Measurement
    engine = db.get_sync_engine()
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql('CREATE SCHEMA clio')
            for model in (File, FileObservation, Measurement):
                model.__table__.create(connection)
        yield engine
    finally:
        db.dispose_sync_engine()




@pytest.fixture
def file_api(file_database, database, tmp_path):
    if shutil.which('go') is None:
        pytest.skip('Go is required')
    _, _, environment = database
    with clio_server(tmp_path, {**environment, 'OBSERVATIONS_SERVICE_TOKEN': 'test-token'}) as url:
        with httpx.Client(base_url=url, headers={'Authorization': 'Bearer test-token'}, trust_env=False) as client:
            yield client
