from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from common.sdo_images import hourly_slots, image_path, prune, read_image, save_image


NOW = datetime(2026, 9, 30, 12, 30, tzinfo=UTC)


def receipt(slot):
    return dict(observed_at=slot.isoformat(), available_at=(slot + timedelta(minutes=20)).isoformat(),
                source='test', preprocessing='test-v1', units='DN', sha256='test-digest')


def test_retention_boundary_and_unrelated_files(tmp_path):
    slots = hourly_slots(NOW)
    assert len(slots) == 1080 and len(set(slots)) == 1080
    image = np.ones((512, 512), dtype=np.float32)
    for slot in (slots[0], slots[-1]):
        assert save_image(tmp_path, slot, 'aia94', image, receipt(slot), now=NOW)
    unrelated = tmp_path / 'notes.txt'
    unrelated.write_text('keep')
    assert prune(tmp_path, now=NOW + timedelta(hours=1)) == 1
    assert image_path(tmp_path, slots[0], 'aia94').exists()
    assert unrelated.exists()
    with pytest.raises(ValueError, match='window'):
        save_image(tmp_path, slots[-1], 'aia94', image, receipt(slots[-1]), now=NOW + timedelta(hours=1))


def test_first_receipt_and_causal_reads(tmp_path):
    slot = hourly_slots(NOW)[0]
    image = np.ones((512, 512), dtype=np.float32)
    assert save_image(tmp_path, slot, 'aia94', image, receipt(slot), now=NOW)
    assert not save_image(tmp_path, slot, 'aia94', image * 2, receipt(slot), now=NOW)
    actual, meta = read_image(tmp_path, slot, 'aia94', as_of=NOW)
    np.testing.assert_array_equal(actual, image)
    assert meta['preprocessing'] == 'test-v1'
    with pytest.raises(ValueError, match='not available'):
        read_image(tmp_path, slot, 'aia94', as_of=slot)
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path, slot, 'hmi_m')


def test_reject_invalid_observations(tmp_path):
    slot = hourly_slots(NOW)[0]
    for image in (np.zeros((256, 256)), np.full((512, 512), np.nan)):
        with pytest.raises(ValueError, match='512x512'):
            save_image(tmp_path, slot, 'aia94', image, receipt(slot), now=NOW)
    with pytest.raises(ValueError, match='timezone'):
        hourly_slots(NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match='whole UTC hour'):
        image_path(tmp_path, NOW, 'aia94')


def test_archive_accepts_only_configured_observation_channels(tmp_path):
    from common.sdo_images import OBSERVED_CHANNELS
    slot = hourly_slots(NOW)[0]
    image = np.ones((512, 512), dtype=np.float32)
    assert len(OBSERVED_CHANNELS) == 9 and {'aia1600', 'hmi_m'}.issubset(OBSERVED_CHANNELS)
    assert save_image(tmp_path, slot, 'hmi_m', image, receipt(slot), now=NOW)
    for channel in ('halpha', 'hmi_v', 'hmi_bx', 'hmi_by', 'hmi_bz'):
        with pytest.raises(ValueError, match='schema/channel'):
            save_image(tmp_path, slot, channel, image, receipt(slot), now=NOW)
    assert prune(tmp_path, now=NOW + timedelta(hours=1080)) == 1


def test_native_original_matches_first_receipt_and_expires_with_image(tmp_path):
    import hashlib
    from common.sdo_images import original_path, save_original
    slot = hourly_slots(NOW)[0]
    content = b'native FITS pixels'
    metadata = {**receipt(slot), 'sha256': hashlib.sha256(content).hexdigest()}
    save_image(tmp_path, slot, 'aia193', np.ones((512, 512), np.float32), metadata, now=NOW)
    with pytest.raises(ValueError, match='SHA256'):
        save_original(tmp_path, slot, 'aia193', b'other pixels', now=NOW)
    assert save_original(tmp_path, slot, 'aia193', content, now=NOW)
    assert not save_original(tmp_path, slot, 'aia193', content, now=NOW)
    assert original_path(tmp_path, slot, 'aia193').read_bytes() == content
    assert prune(tmp_path, now=NOW + timedelta(hours=1080)) == 1
    assert not original_path(tmp_path, slot, 'aia193').exists()
