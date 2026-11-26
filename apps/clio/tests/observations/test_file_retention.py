from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from clio.config import load_observation_config
from clio.db.locks import FILE_LOCKS, ingestion_lock
from clio.ingestion.cleanup import cleanup_files
from clio.ingestion.utils.file_observations import find, save

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)


def stored(observation, at, *, content=b'original', variants=False):
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'original.fits'
        path.write_bytes(content)
        variant = path.with_name('512.webp')
        if variants:
            variant.write_bytes(b'variant')
        return save(path, observation=observation, observed_at=at, source_product='source',
                    variants=[variant] if variants else [])


def test_retention_is_per_observation_and_unknown_observations_are_preserved(storage):
    config = load_observation_config()
    config.observations = {name: config.observations[name] for name in ('aia_171_nrt', 'aia_211_nrt')}
    config.observations['aia_171_nrt'].retention_days = 1
    config.observations['aia_211_nrt'].retention_days = 2
    stored('aia_171_nrt', NOW-timedelta(days=1))
    stored('aia_171_nrt', NOW-timedelta(days=1, seconds=1))
    stored('aia_211_nrt', NOW-timedelta(days=1, hours=1))
    stored('unknown', NOW-timedelta(days=10))
    report = cleanup_files(config, NOW)
    assert report['aia_171_nrt']['expired'] == 1
    assert report['aia_211_nrt']['expired'] == 0
    assert len(find('aia_171_nrt')) == len(find('unknown')) == 1
    assert len(storage.objects) == 3


def test_partial_delete_hides_incomplete_file_and_retries(storage, monkeypatch):
    config = load_observation_config()
    config.observations = {'gong': config.observations['gong']}
    row = stored('gong', NOW-timedelta(days=90), variants=True)
    delete = storage.client.delete_object.side_effect

    def fail_original(**kwargs):
        if kwargs['Key'] == row['key']:
            raise OSError('S3 unavailable')
        return delete(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.client, 'delete_object', Mock(side_effect=fail_original))
        assert cleanup_files(config, NOW)['gong']['failed'] == 1
    assert not find('gong') and len(storage.objects) == 1
    assert cleanup_files(config, NOW)['gong']['expired'] == 1
    assert not find('gong') and not storage.objects


def test_cleanup_obeys_collection_lock_and_batch_limit(storage):
    config = load_observation_config()
    config.observations = {'gong': config.observations['gong']}
    for day in (90, 91):
        stored('gong', NOW-timedelta(days=day))
    with ingestion_lock('gong', FILE_LOCKS['gong']):
        assert cleanup_files(config, NOW)['gong']['busy']
    assert cleanup_files(config, NOW, max_files=1)['gong']['expired'] == 1
    assert len(find('gong')) == 1


def test_replaying_identical_original_reuses_receipt(storage):
    first = stored('gong', NOW)
    replay = stored('gong', NOW)
    assert replay['id'] == first['id']
    assert replay['file']['timestamp'] == first['file']['timestamp']
    assert len(find('gong')) == len(storage.objects) == 1
