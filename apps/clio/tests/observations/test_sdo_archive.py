"""Validate the scientific SDO contract independently of its storage backend."""
from datetime import UTC, datetime, timedelta
import numpy as np
import pytest
from clio.observations import sdo

NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def test_hourly_collection_window():
    slots = sdo.hourly_slots(NOW)
    assert len(slots) == 96 and len(set(slots)) == 96
    assert slots[0] == NOW.replace(minute=0) and slots[-1] == slots[0]-timedelta(hours=95)


def test_metadata_and_pixel_validation(scientific_source):
    from clio.providers.sdo import prepare_observation
    image, metadata = prepare_observation(scientific_source.image, scientific_source.download.return_value[1])
    metadata.update(version=1, slot_at=scientific_source.slot.isoformat())
    sdo.validate(image, metadata)
    with pytest.raises(ValueError):
        sdo.validate(image.astype(np.float64), metadata)
    image[0, 0] = np.inf
    with pytest.raises(ValueError):
        sdo.validate(image, metadata)


def test_object_keys_do_not_include_local_paths():
    assert sdo.object_key('aia193', NOW, 'npz.512') == 'images/SDO/aia193/202611021030/512.npz'
    assert sdo.object_key('aia193', NOW, 'jpg.original', public=True) == 'images/AIA/193/202611021030/original.jpg'
    with pytest.raises(ValueError):
        sdo.object_key('other', NOW, 'npz.512')
