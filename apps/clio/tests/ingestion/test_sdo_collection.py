from datetime import timedelta
from unittest.mock import Mock
import numpy as np
import pytest
import requests

from clio.ingestion.extras import sdo as collector
from clio.db.locks import ingestion_lock
from clio.ingestion.extras import sdo
from clio.ingestion.utils import file_observations
from clio.providers.sdo import observation_url, prepare_observation


@pytest.fixture
def policy():
    from clio.config import load_observation_config
    return load_observation_config().observations['aia_171_nrt'].model_copy(update={'freshness_minutes': 60})


def collect(policy, now, mode='live'):
    return collector.collect(policy, channel='aia171', source='aia.nrt_171', mode=mode, now=now)


def test_live_collection_is_idempotent(storage, scientific_source, policy):
    first = collect(policy, scientific_source.now)
    assert first['saved'] == 1 and not first['failed']
    before = dict(storage.objects)
    assert collect(policy, scientific_source.now)['existing'] == 1
    assert before == storage.objects and scientific_source.download.call_count == 1


def test_duplicate_lane_is_skipped(storage, policy, scientific_source):
    with ingestion_lock('sdo-live', 730200):
        assert collect(policy, scientific_source.now)['status'] == 'busy'


def test_backfill_uses_policy_window_and_channel(storage, policy, scientific_source, monkeypatch):
    policy = policy.model_copy(update={'retention_days': 2, 'backfill_days': 1})
    collect_one = Mock(return_value='existing')
    monkeypatch.setattr(collector, 'collect_one', collect_one)
    report = collector.collect(policy, channel='aia1600', source='aia.nrt_1600', mode='backfill', now=scientific_source.now)
    assert report['existing'] == 24
    assert {call.args[1] for call in collect_one.call_args_list} == {'aia1600'}
    assert all(call.args[0] >= scientific_source.now-timedelta(days=1) for call in collect_one.call_args_list)


def test_requested_slots_stay_fixed_while_download_time_advances(storage, scientific_source):
    clocks = iter([scientific_source.now, scientific_source.now+timedelta(minutes=1)])
    assert collector.collect_one(scientific_source.slot, 'aia171', clock=lambda: next(clocks)) == 'saved'
    metadata = file_observations.find(entity='npz.512')[0]['details']
    assert metadata['available_at'] == (scientific_source.now+timedelta(minutes=1)).isoformat()


def test_404_does_not_publish_objects(storage, scientific_source):
    response = requests.Response()
    response.status_code = 404
    scientific_source.download.side_effect = requests.HTTPError(response=response)
    assert collector.collect_one(scientific_source.slot, 'aia171', clock=lambda: scientific_source.now) == 'unavailable'
    assert not storage.objects


def test_reduction_retains_nan_and_rescales_wcs(scientific_source):
    image, metadata = prepare_observation(scientific_source.image, scientific_source.download.return_value[1])
    assert image.shape == (512, 512) and np.isnan(image[0, 0])
    assert metadata['header']['CRPIX1'] == 256.5 and metadata['header']['CDELT1'] == 4.
    assert observation_url('aia171', scientific_source.slot).endswith('_0171.fits')


def test_unknown_channel_is_rejected_before_download(scientific_source):
    assert collector.collect_one(scientific_source.slot, 'other', clock=lambda: scientific_source.now) == 'unavailable'
    scientific_source.download.assert_not_called()


def test_sdo_configuration_uses_independent_product_policies():
    from clio.config import ClioObservations, load_observation_config
    from pydantic import ValidationError
    config = load_observation_config()
    policy = config.observations['aia_171_nrt'].model_dump()
    policy.update(retention_days=45, backfill_days=45)
    assert ClioObservations(observations={'aia_171_nrt': policy}).observations['aia_171_nrt'].retention_days == 45
    with pytest.raises(ValidationError, match='backfill exceeds retention'):
        ClioObservations(observations={'aia_171_nrt': {**policy, 'retention_days': 4}})
    with pytest.raises(ValidationError, match='incompatible'):
        ClioObservations(observations={'aia_171_nrt': {**policy, 'backfill_days': 0, 'sources': {'live': ['nasa.aia_browse'], 'historical': []}}})
    with pytest.raises(ValidationError):
        ClioObservations(observations={}, sdo_images={'enabled': True})
