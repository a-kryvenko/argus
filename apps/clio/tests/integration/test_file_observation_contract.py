"""Upgrade the deployed file schema, then exercise real collection writes."""
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
import sqlalchemy as sa
from sqlalchemy.engine import make_url

from tests.support.database import database


def test_deployed_schema_preserves_receipts_and_accepts_current_collection(database, object_store, monkeypatch, tmp_path):
    _, urls, environment = database
    from clio.db import session as db
    from clio.db.base import Base
    from clio.ingestion.utils.file_observations import find, save

    engine = sa.create_engine(make_url(urls['clio']).set(drivername='postgresql+psycopg'),
                              connect_args={'options': '-csearch_path=clio,public'})
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[2] / 'src/clio/db/migrations'))
    now = datetime(2026, 11, 7, 12, tzinfo=UTC)
    original_id, duplicate_id, derivative_id = uuid4(), uuid4(), uuid4()
    file_ids = [uuid4() for _ in range(7)]
    originals = [
        ('sdo', 'aia171', 'fits.original', 'aia.nrt_171', 'aia_171_nrt'),
        ('sdo_browse', 'aia193', 'jpg.original', 'nasa.aia_browse', 'aia_193_browse'),
        ('gong', 'gong', 'fits.original', 'gong.live', 'gong'),
        ('goes', 'goes', 'json.original', 'goes.live', 'goes'),
        ('donki', None, 'json.original', 'CME', 'donki.CME'),
        ('enlil', None, 'netcdf.original', 'noaa.enlil', 'enlil'),
    ]
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql('CREATE SCHEMA clio')
            context = MigrationContext.configure(connection, opts={'target_metadata': Base.metadata})
            with Operations.context(context):
                scripts.get_revision('20261027_file_observation').module.upgrade()
                scripts.get_revision('20261102_files').module.upgrade()
            for file_id in file_ids:
                connection.execute(sa.text('INSERT INTO clio.files VALUES (:id,:time,\'argus-private\',:key)'),
                                   dict(id=file_id, time=now, key=str(file_id)))
            for index, (kind, variable, entity, source, _) in enumerate(originals):
                connection.execute(sa.text('''INSERT INTO clio.file_observation
                    (id,kind,source_product,source_key,received_at,file_id,variable,observed_at,entity,details)
                    VALUES (:id,:kind,:source,:key,:time,:file,:variable,:observed,:entity,
                            jsonb_build_object('sha256', CAST(:checksum AS text), 'complete', true))'''),
                    dict(id=original_id if index == 4 else uuid4(), kind=kind, source=source, key=str(index),
                         time=now, file=file_ids[index], variable=variable, observed=now if variable else None,
                         entity=entity, checksum='a'*64))
            # Old poll receipts may share an original. Derivatives are separate files.
            connection.execute(sa.text('''INSERT INTO clio.file_observation
                (id,kind,source_product,source_key,received_at,file_id,entity,details)
                SELECT :id,kind,source_product,'duplicate',received_at,file_id,entity,details
                FROM clio.file_observation WHERE id=:original'''), dict(id=duplicate_id, original=original_id))
            connection.execute(sa.text('''INSERT INTO clio.file_observation
                (id,kind,source_product,source_key,received_at,file_id,variable,observed_at,entity,details)
                VALUES (:id,'sdo_browse','nasa.aia_browse','derivative',:time,:file,'aia193',:time,
                        'webp.512',jsonb_build_object('sha256', CAST(:checksum AS text), 'complete', true))'''),
                dict(id=derivative_id, time=now, file=file_ids[6], checksum='a'*64))
            connection.execute(sa.text('''INSERT INTO clio.file_observation
                (id,kind,source_product,source_key,received_at,raw_path,sha256)
                VALUES (:id,'goes','goes.archive','local',:time,'legacy.gz',:checksum)'''),
                dict(id=uuid4(), time=now, checksum='b'*64))
            connection.exec_driver_sql('CREATE TABLE clio.alembic_version (version_num varchar(32) PRIMARY KEY)')
            connection.exec_driver_sql("INSERT INTO clio.alembic_version VALUES ('20261102_files')")

        command = [sys.executable, '-m', 'alembic', '-c',
                   str(Path(__file__).resolve().parents[2] / 'alembic.ini'), 'upgrade', 'head']
        for _ in range(2):
            result = subprocess.run(command, env={**environment, 'PYTHONPATH': ''}, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr

        monkeypatch.setattr(db, 'get_sync_engine', lambda: engine)
        for _, _, _, source, observation in originals:
            records = find(observation)
            assert len(records) == 1 and records[0]['source_product'] == source
            assert records[0]['sha256'] == 'a'*64 and records[0]['observed_at'] == now
        with engine.connect() as connection:
            assert connection.scalar(sa.text('SELECT version_num FROM clio.alembic_version')) == '20261107_event_identities'
            assert connection.scalar(sa.text('SELECT count(*) FROM clio.file_observation')) == 9
            assert connection.scalar(sa.text('SELECT count(*) FROM clio.file_observation WHERE observation IS NULL')) == 3
            assert connection.scalar(sa.text("SELECT raw_path FROM clio.file_observation WHERE source_key='local'")) == 'legacy.gz'

        path = tmp_path/'original.fits'
        path.write_bytes(b'new original')
        saved = save(path, observation='aia_171_nrt', observed_at=now, source_product='aia.nrt_171')
        assert saved['id'] in {row['id'] for row in find('aia_171_nrt')}

        from clio.ingestion.extras.donki import record_poll
        raw = json.dumps([dict(activityID='cme1', startTime=now.isoformat())]).encode()
        metadata = dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1))
        for minute in range(2):
            record_poll('CME', raw=raw, metadata=metadata, observed_at=now+timedelta(minutes=minute))
            with engine.connect() as connection:
                assert connection.scalar(sa.text("SELECT count(*) FROM clio.file_observation WHERE observation='donki.CME'")) == 2
                assert connection.scalar(sa.text("SELECT count(*) FROM clio.file_observation WHERE observation='donki.CME' AND source_id='cme1'")) == 1
    finally:
        engine.dispose()
