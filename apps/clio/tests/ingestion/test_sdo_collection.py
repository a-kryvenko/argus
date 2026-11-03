from datetime import timedelta
from unittest.mock import Mock
import numpy as np
import pytest
import requests

from clio.ingestion import sdo_images as collector
from clio.db.locks import ingestion_lock
from clio.observations import sdo, file_observations
from clio.providers.sdo import observation_url, prepare_observation


def test_live_collection_is_idempotent(storage, scientific_source, monkeypatch):
    monkeypatch.setattr(collector, 'COLLECTED_CHANNELS', ('aia171',))
    monkeypatch.setattr(collector, 'hourly_slots', lambda now: [scientific_source.slot])
    first = collector.cycle('live', clock=lambda: scientific_source.now)
    assert first['saved'] == 1 and not first['failed']
    before = dict(storage.objects)
    assert collector.cycle('live', clock=lambda: scientific_source.now)['existing'] == 1
    assert before == storage.objects and scientific_source.download.call_count == 1


def test_duplicate_lane_is_skipped(storage):
    with ingestion_lock('sdo-live', 730200):
        assert collector.cycle('live') == {'busy': True}


def test_warmup_rotates_past_unavailable_sources(storage, scientific_source, monkeypatch):
    monkeypatch.setattr(collector, 'COLLECTED_CHANNELS', ('aia171',))
    monkeypatch.setattr(collector, '_warmup_cursor', None)
    visited = []
    def collect(slot, channel, **kwargs):
        visited.append(slot)
        return 'unavailable'
    monkeypatch.setattr(collector, 'collect_one', collect)
    for _ in range(3):
        assert collector.cycle('warmup', batch_hours=2, now=scientific_source.now)['unavailable'] == 2
    assert len(set(visited)) == 6


def test_requested_slots_stay_fixed_while_download_time_advances(storage, scientific_source, monkeypatch):
    monkeypatch.setattr(collector, 'COLLECTED_CHANNELS', ('aia171',))
    monkeypatch.setattr(collector, 'hourly_slots', lambda now: [scientific_source.slot])
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


def test_full_warmup_only_collects_proswin_channels_for_96_hours(storage, scientific_source, monkeypatch):
    collect = Mock(return_value='existing')
    monkeypatch.setattr(collector, 'collect_one', collect)
    report = collector.cycle('warmup', full=True, now=scientific_source.now)
    assert report['existing'] == 192
    assert {call.args[1] for call in collect.call_args_list} == {'aia171', 'aia211'}
    slots = {call.args[0] for call in collect.call_args_list}
    assert len(slots) == 96
    assert min(slots) == scientific_source.slot - timedelta(hours=95)
    assert max(slots) == scientific_source.slot


@pytest.mark.parametrize('channel, hours, status', [('aia193', 0, 'unavailable'), ('aia171', 96, 'expired')])
def test_unneeded_observations_are_rejected_before_download(scientific_source, channel, hours, status):
    assert collector.collect_one(scientific_source.slot-timedelta(hours=hours), channel,
        clock=lambda: scientific_source.now) == status
    scientific_source.download.assert_not_called()


def test_sdo_configuration_rejects_a_45_day_backfill():
    from clio.config import SDOImages
    from pydantic import ValidationError
    assert SDOImages().backfill_days == 4
    with pytest.raises(ValidationError):
        SDOImages(backfill_days=45)
