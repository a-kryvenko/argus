import pytest
from tests.support.database import database


@pytest.fixture
def file_database(database, monkeypatch, object_store):
    _, _, environment = database
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db import session as db
    from clio.db.models.file import File
    from clio.db.models.file_observation import FileObservation
    from clio.db.models.measurement import Measurement
    engine = db.get_sync_engine()
    for model in (File, FileObservation, Measurement):
        model.__table__.create(engine)
    yield engine


