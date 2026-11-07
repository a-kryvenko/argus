import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from clio.config import ClioObservations, load_observation_config
from clio.ingestion.extras import sdo, sdo_browse
from clio.ingestion.utils.file_observations import save
from tempfile import TemporaryDirectory
from pathlib import Path
from clio.ingestion.utils.file_collection import collect_observation

NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def seed(storage, observation, observed, *, complete=True):
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'original.fits'
        path.write_bytes(b'original')
        save(path, observation=observation, observed_at=NOW if observation.endswith('_browse') else observed,
             source_product='nasa.aia_browse' if observation.endswith('_browse') else 'aia.nrt_171',
             details={'source_updated_at': observed.isoformat() if observed else None})


@pytest.mark.parametrize('observation,module,age,expected', [
    (observation, module, age, expected)
    for observation, module in [('aia_171_nrt', sdo), ('aia_171_browse', sdo_browse)]
    for age, expected in [(30, 'fresh'), (180, 'stale'), (None, 'unknown')]
    if age is not None or observation.endswith('_browse')])
def test_live_freshness_uses_source_time_not_upload_time(storage, monkeypatch, observation, module, age, expected):
    config = load_observation_config()
    policy = config.observations[observation]
    seed(storage, observation, NOW-timedelta(minutes=age) if age is not None else None)
    collect = Mock(return_value={'status': 'complete', 'failed': 0})
    monkeypatch.setattr(module, 'collect', collect)
    report = asyncio.run(collect_observation(config, observation, mode='live', now=NOW))
    assert report['freshness'] == expected
    assert report['status'] == ('complete' if expected == 'fresh' else 'partial')
    assert collect.call_args.args == (policy,)
    assert collect.call_args.kwargs['channel'] == 'aia171'


def test_missing_scientific_observation_time_is_rejected(storage):
    with pytest.raises(ValueError, match='Observation time'):
        seed(storage, 'aia_171_nrt', None)


def test_backfill_does_not_require_old_observations_to_be_fresh(storage, monkeypatch):
    seed(storage, 'aia_171_nrt', NOW-timedelta(days=2))
    monkeypatch.setattr(sdo, 'collect', Mock(return_value={'status': 'complete', 'failed': 0}))
    report = asyncio.run(collect_observation(load_observation_config(), 'aia_171_nrt', mode='backfill', now=NOW))
    assert report['freshness'] == 'stale' and report['status'] == 'complete'


def test_browse_without_history_is_skipped_before_downloading(monkeypatch):
    collect = Mock(side_effect=AssertionError('No historical browse source'))
    monkeypatch.setattr(sdo_browse, 'collect', collect)
    report = asyncio.run(collect_observation(load_observation_config(), 'aia_171_browse', mode='backfill', now=NOW))
    assert report['status'] == 'skipped'
    collect.assert_not_called()


def test_nrt_products_can_be_configured_per_channel():
    policy = load_observation_config().observations['aia_171_nrt'].model_dump()
    policy['sources'] = {'live': ['aia.nrt_94'], 'historical': ['aia.nrt_94']}
    config = ClioObservations(observations={'aia_094_nrt': policy})
    assert list(config.observations) == ['aia_094_nrt']
