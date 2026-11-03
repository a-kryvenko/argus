from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from astropy.io import fits

from clio.providers.sdo import fits_observed_at, read_native_fits


SLOT = datetime(2026, 9, 30, 6, tzinfo=UTC)


def test_hmi_tai_is_not_mistaken_for_utc():
    observed = fits_observed_at({'T_OBS': '2026.09.30_06:00:00.000_TAI'})
    assert observed == SLOT - timedelta(seconds=37)


def test_date_and_time_are_combined():
    assert fits_observed_at({'DATE-OBS': '2026-09-30', 'TIME-OBS': '06:00:00'}) == SLOT


def test_stale_current_image_is_not_a_live_observation(tmp_path):
    path = tmp_path / 'current.fts'
    header = fits.Header({'DATE_OBS': '2023-07-11T18:07:07', 'INSTRUME': 'AIA_1', 'WAVELNTH': 171})
    fits.writeto(path, np.ones((1024, 1024), dtype=np.float32), header)
    with pytest.raises(ValueError, match='Stale'):
        read_native_fits(path, 'aia171', slot=SLOT, now=SLOT + timedelta(hours=1))


def test_nrt_fits_keeps_resolution_and_is_not_labelled_encoder_ready(tmp_path):
    path = tmp_path / 'aia.fits'
    header = fits.Header({'DATE-OBS': SLOT.isoformat().replace('+00:00', ''),
                          'INSTRUME': 'AIA_1', 'WAVELNTH': 171})
    fits.writeto(path, np.ones((1024, 1024), dtype=np.float32), header)
    data, metadata = read_native_fits(path, 'aia171', slot=SLOT, now=SLOT + timedelta(hours=1))
    assert data.shape == (1024, 1024)
    assert metadata['native_shape'] == [1024, 1024]
    assert metadata['preprocessing'] == 'fits-unregistered-v1'


def test_los_keeps_source_pixels_and_provenance(tmp_path):
    path = tmp_path / 'los.fts'
    header = fits.Header({'DATE-OBS': SLOT.isoformat().replace('+00:00', ''), 'INSTRUME': 'HMI', 'BUNIT': 'Gauss'})
    values = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    fits.writeto(path, values, header)
    data, metadata = read_native_fits(path, 'hmi_m', slot=SLOT, now=SLOT + timedelta(minutes=1))
    np.testing.assert_array_equal(data, values)
    assert metadata['native_shape'] == [1024, 1024] and len(metadata['sha256']) == 64


@pytest.mark.parametrize('channel, flags, accepted', [
    ('aia171', 1, False), ('aia171', 1 << 30, True), ('hmi_m', 1 << 30, False),
])
def test_fits_flags_are_checked_before_returning_pixels(tmp_path, channel, flags, accepted):
    path = tmp_path / 'flagged.fits'
    header = fits.Header({'DATE-OBS': SLOT.isoformat().replace('+00:00', ''), 'INSTRUME': 'AIA' if channel == 'aia171' else 'HMI',
                          'WAVELNTH': 171, 'BUNIT': 'Gauss', 'QUALITY': flags})
    fits.writeto(path, np.ones((1024, 1024), dtype=np.float32), header)
    if accepted:
        _, metadata = read_native_fits(path, channel, slot=SLOT, now=SLOT + timedelta(minutes=1))
        assert 'quality' not in metadata
    else:
        with pytest.raises(ValueError, match='quality flags'):
            read_native_fits(path, channel, slot=SLOT, now=SLOT + timedelta(minutes=1))


def test_file_reference_uses_internal_channel_name():
    from clio.providers.sdo import fetch
    result = fetch('aia171', SLOT)
    assert set(result) == {'timestamp', 'aia171_url'}
    assert result['timestamp'] == SLOT
    assert result['aia171_url'].endswith('AIA20260930_060000_0171.fits')


def test_provider_returns_reduced_observation_and_unchanged_original(scientific_source):
    from clio.providers import sdo
    source = scientific_source
    image, metadata, original = sdo.load_observation('aia171', source.slot, now=source.now)
    assert image.shape == (512, 512) and image.dtype == np.float32
    assert metadata['header']['CRPIX1'] == 256.5
    assert original == source.content
    source.download.assert_called_once_with(sdo.observation_url('aia171', source.slot), 'aia171',
        slot=source.slot, now=source.now, include_original=True)


@pytest.mark.parametrize('limit, accepted', [(16, True), (3, False)])
def test_browse_download_is_bounded_and_preserves_source_time(monkeypatch, limit, accepted):
    from unittest.mock import MagicMock
    from clio.providers import sdo
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [b'jpeg']
    response.headers = {'last-modified': 'Wed, 30 Sep 2026 06:00:00 GMT'}
    monkeypatch.setattr(sdo.requests, 'get', response)
    response.return_value = response
    monkeypatch.setattr(sdo, 'MAX_BROWSE_BYTES', limit)
    if accepted:
        assert sdo.download_browse('193') == (b'jpeg', SLOT)
    else:
        with pytest.raises(ValueError, match='size/time limit'):
            sdo.download_browse('193')
