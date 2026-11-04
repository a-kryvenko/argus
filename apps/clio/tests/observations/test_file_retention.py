from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select

from clio.config import load_observation_config
from clio.db.locks import FILE_LOCKS, ingestion_lock
from clio.db.models.file_observation import FileObservation as Receipt
from clio.ingestion.utils import file_observations
from clio.ingestion import cleanup as retention

NOW = datetime(2026, 11, 3, 12, 30, tzinfo=UTC)


def snapshot(storage, kind, at, *, variable=None, entity=None, details=None, reference=None):
    storage.clock[0] = at
    bucket, prefix = {
        'sdo': ('argus-private', 'images/SDO/'),
        'sdo_browse': ('argus-public', 'images/AIA/'),
        'gong': ('argus-private', 'observations/GONG/'),
        'goes': ('argus-private', 'observations/GOES/'),
        'enlil': ('argus-private', 'observations/ENLIL/'),
    }[kind]
    reference = reference or storage.publish(b'original', bucket, prefix+uuid4().hex,
                                                  content_type='application/octet-stream')
    file_observations.register(variable=variable or kind, timestamp=at, entity=entity or 'fits.original',
        bucket=bucket, key=reference['key'], uploaded_at=reference['timestamp'], kind=kind,
        source_product=kind+'.test', details=details)
    return reference


def remaining(storage, kind):
    with storage.engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(Receipt.__table__).where(Receipt.kind == kind)).mappings()]


def test_sdo_respects_independent_channel_retention_and_shared_objects(storage):
    config = load_observation_config()
    config.observations = {m: config.observations[m] for m in ('aia_171_nrt', 'aia_211_nrt')}
    config.observations['aia_171_nrt'].retention_days = 1
    config.observations['aia_211_nrt'].retention_days = 2
    boundary = snapshot(storage, 'sdo', NOW-timedelta(days=1), variable='aia171')
    recent = snapshot(storage, 'sdo', NOW-timedelta(days=1, hours=12), variable='aia211')
    snapshot(storage, 'sdo', NOW-timedelta(days=1, minutes=1), variable='aia171')
    snapshot(storage, 'sdo', NOW-timedelta(days=2), variable='aia171', reference=boundary)
    report = retention.cleanup_files(config, NOW)
    assert report['aia_171_nrt'] == dict(expired=2, files_deleted=1, failed=0)
    assert report['aia_211_nrt']['expired'] == 0
    assert {row['file_id'] for row in remaining(storage, 'sdo')} == {boundary['id'], recent['id']}


def test_gong_and_goes_respect_configured_hour_and_day_boundaries(storage):
    config = load_observation_config()
    for kind in ('gong', 'goes'):
        lower = NOW-timedelta(days=config.observations[kind].retention_days)
        snapshot(storage, kind, lower)
        snapshot(storage, kind, lower-timedelta(minutes=1))
    report = retention.cleanup_files(config, NOW)
    for kind in ('gong', 'goes'):
        assert report[kind] == dict(expired=1, files_deleted=1, failed=0)
        assert len(remaining(storage, kind)) == 1


def test_public_image_retention_is_per_product_and_not_an_implicit_latest_exemption(storage):
    config = load_observation_config()
    config.observations = {m: config.observations[m] for m in ('aia_193_browse', 'aia_171_browse')}
    config.observations['aia_171_browse'].retention_days = 2
    snapshot(storage, 'sdo_browse', NOW-timedelta(hours=25), variable='aia193',
             entity='jpg.original', details={'complete': True})
    snapshot(storage, 'sdo_browse', NOW-timedelta(hours=2), variable='aia193',
             entity='jpg.original', details={'complete': True})
    snapshot(storage, 'sdo_browse', NOW-timedelta(hours=1), variable='aia193',
             entity='jpg.original', details={'complete': False})
    snapshot(storage, 'sdo_browse', NOW-timedelta(hours=36), variable='aia171',
             entity='jpg.original', details={'complete': True})
    report = retention.cleanup_files(config, NOW)
    assert report['aia_193_browse'] == dict(expired=1, files_deleted=1, failed=0)
    assert report['aia_171_browse']['expired'] == 0
    assert len(remaining(storage, 'sdo_browse')) == 3


def test_enlil_keeps_model_selected_at_retention_boundary_even_if_older_run_arrived_later(storage):
    def profile(at, created):
        return snapshot(storage, 'enlil', at, details={
            'model_created_at': created.isoformat(), 'source_modified_at': created.isoformat()})
    profile(NOW-timedelta(days=50), NOW-timedelta(days=51))
    baseline = profile(NOW-timedelta(days=33), NOW-timedelta(days=34))
    profile(NOW-timedelta(days=32), NOW-timedelta(days=40))
    # Most recent attempt before the boundary is retained independently of model selection.
    last_attempt = profile(NOW-timedelta(days=31), NOW-timedelta(days=39))
    current = profile(NOW, NOW-timedelta(hours=1))
    report = retention.cleanup_files(load_observation_config(), NOW)
    assert report['enlil'] == dict(expired=2, files_deleted=2, failed=0)
    assert {row['file_id'] for row in remaining(storage, 'enlil')} == {baseline['id'], last_attempt['id'], current['id']}


def test_cleanup_skips_active_collection_and_retries_after_lock_release(storage):
    snapshot(storage, 'sdo', NOW-timedelta(days=5), variable='aia171')
    with ingestion_lock('sdo-warmup', FILE_LOCKS['sdo-warmup']):
        assert retention.cleanup_files(load_observation_config(), NOW)['aia_171_nrt']['busy']
        assert len(storage.objects) == 1
    assert retention.cleanup_files(load_observation_config(), NOW)['aia_171_nrt']['files_deleted'] == 1


def test_delete_batch_is_bounded_and_unknown_or_referenced_prefixes_are_preserved(storage):
    for n in range(3):
        storage.publish(b'orphan', 'argus-private', f'observations/GONG/unused-{n}',
                             content_type='application/octet-stream')
    other = storage.publish(b'unmanaged', 'argus-private', 'other/keep', content_type='application/octet-stream')
    report = retention.cleanup_files(load_observation_config(), NOW, max_files=2)
    assert report['gong']['files_deleted'] == 2
    assert len(storage.objects) == 2
    assert retention.cleanup_files(load_observation_config(), NOW)['gong']['files_deleted'] == 1
    assert storage.files(id=other['id'])


def test_recent_unreferenced_upload_gets_a_grace_period_for_interrupted_publication(storage):
    storage.clock[0] = NOW
    reference = storage.publish(b'pending', 'argus-private', 'images/SDO/pending.npz',
                                     content_type='application/octet-stream')
    assert retention.cleanup_files(load_observation_config(), NOW)['aia_171_nrt']['files_deleted'] == 0
    assert storage.files(id=reference['id'])
    assert retention.cleanup_files(load_observation_config(), NOW+timedelta(minutes=16))['aia_171_nrt']['files_deleted'] == 1
