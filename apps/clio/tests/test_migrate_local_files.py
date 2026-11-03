from datetime import UTC, datetime, timedelta
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest
from sqlalchemy import insert, select

from clio import migrate_local_files as migration
from clio.db.models.file_observation import FileObservation
from clio.observations import files, file_observations, sdo, sdo_browse

SLOT = datetime(2026,11,2,10,tzinfo=UTC)
NOW = SLOT+timedelta(minutes=30)


def seed_receipt(storage, monkeypatch, kind):
    root = storage.root/kind
    root.mkdir()
    monkeypatch.setenv(f'ARGUS_{kind.upper()}_ARCHIVE', str(root))
    raw = b'[]' if kind == 'donki' else b'original'
    digest = hashlib.sha256(raw).hexdigest()
    record = dict(slot_at=SLOT.isoformat(), observed_at=SLOT.isoformat(), available_at=NOW.isoformat(),
        sha256=digest, source_product=kind+'.live', source_url='https://example.test/source')
    path = root/'original.gz'
    path.write_bytes(gzip.compress(raw))
    path.with_suffix('.gz.receipt.json').write_text(json.dumps(record))
    values = dict(id=uuid4(), kind=kind, source_product=kind+'.live', source_key='legacy-'+kind,
        received_at=NOW, raw_path='original.gz', sha256=digest)
    with storage.engine.begin() as connection:
        connection.execute(insert(FileObservation).values(**values))
    return root, values, raw


@pytest.mark.parametrize('kind', ['gong','goes','donki','enlil'])
def test_one_time_migration_is_read_only_until_apply_and_preserves_observation_id(storage, monkeypatch, kind):
    root, values, raw = seed_receipt(storage,monkeypatch,kind)
    before = {str(path):path.read_bytes() for path in root.iterdir()}
    dry = migration.run(kinds=[kind])
    assert not dry['errors'] and not storage.objects and not files.find()
    for _ in range(2):
        report = migration.run(kinds=[kind],apply=True)
        assert not report['errors'], report
    assert before == {str(path):path.read_bytes() for path in root.iterdir()}
    row = file_observations.find(kind=kind)[0]
    assert row['id'] == values['id']
    assert storage.objects[row['file']['bucket'],row['file']['key']][0] == raw
    assert len(storage.objects) == 1


def test_migration_uploads_scientific_arrays_and_legacy_native_fits(storage,scientific_source,monkeypatch):
    from clio.providers.sdo import prepare_observation
    root = storage.root/'sdo'
    directory = root/f'{SLOT:%Y%m%dT%H0000Z}'
    directory.mkdir(parents=True)
    monkeypatch.setenv('ARGUS_SDO_ARCHIVE',str(root))
    image, metadata = prepare_observation(scientific_source.image,scientific_source.download.return_value[1])
    metadata.update(version=1,slot_at=SLOT.isoformat())
    np.savez_compressed(directory/'aia171.npz',image=image,metadata=json.dumps(metadata))
    (directory/'aia171.fits').write_bytes(scientific_source.content)
    # These files must be excluded without opening or uploading their content.
    for channel, slot in [('aia193', SLOT), ('aia171', SLOT-timedelta(hours=96)),
                          ('aia171', SLOT+timedelta(hours=1))]:
        skipped = root/f'{slot:%Y%m%dT%H0000Z}'/(channel+'.npz')
        skipped.parent.mkdir(parents=True, exist_ok=True)
        skipped.write_bytes(b'not an NPZ; outside the active collection')
    assert not migration.run(kinds=['sdo'],now=NOW)['errors'] and not storage.objects
    report = migration.run(kinds=['sdo'],apply=True,now=NOW)
    assert not report['errors'],report
    assert report['sources']['sdo']['scanned'] == 4
    assert report['sources']['sdo']['completed'] == 1
    assert report['sources']['sdo']['skipped'] == 3
    limited = migration.run(kinds=['sdo'],apply=True,now=NOW,limit=1)
    assert limited['sources']['sdo']['completed'] == 1
    assert not limited['errors']
    assert {row['entity'] for row in file_observations.find()} == {'npz.512','fits.512','fits.original'}
    assert (directory/'aia171.npz').exists()


def test_migration_builds_public_variants_from_old_jpeg(storage,monkeypatch):
    root = storage.root/'browse'
    root.mkdir()
    monkeypatch.setenv('ARGUS_SDO_BROWSE_ARCHIVE',str(root))
    monkeypatch.setattr(sdo_browse,'SIZES',(128,512,4096))
    output = io.BytesIO()
    Image.new('RGB',(4096,4096),'red').save(output,'JPEG')
    raw = output.getvalue()
    digest = hashlib.sha256(b'sdo-browse-v1\0'+raw).hexdigest()
    asset = '2026110210_193_'+digest
    (root/(asset+'_4096.jpg')).write_bytes(raw)
    (root/(asset+'.json')).write_text(json.dumps(dict(channel='193',asset_id=asset,
        captured_at=NOW.isoformat(),source_updated_at=None)))
    assert not migration.run(kinds=['sdo_browse'],apply=True)['errors']
    assert len(storage.objects) == 6 and sdo_browse.catalog().images[0].asset_id == asset


def test_corrupted_legacy_file_is_reported_without_upload(storage,monkeypatch):
    root,_,_ = seed_receipt(storage,monkeypatch,'gong')
    (root/'original.gz').write_bytes(gzip.compress(b'corrupted'))
    report = migration.run(kinds=['gong'],apply=True)
    assert report['sources']['gong']['failed'] == 1 and not storage.objects
