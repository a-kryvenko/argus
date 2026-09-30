from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from astropy.io import fits

from clio.providers.sdo_images import fits_observed_at, read_native_fits


SLOT = datetime(2026, 9, 30, 6, tzinfo=UTC)


def test_hmi_tai_is_not_mistaken_for_utc():
    observed = fits_observed_at({'T_OBS': '2026.09.30_06:00:00.000_TAI'})
    assert observed == SLOT - timedelta(seconds=37)


def test_date_and_time_are_combined():
    assert fits_observed_at({'DATE-OBS': '2026-09-30', 'TIME-OBS': '06:00:00'}) == SLOT


def test_stale_current_image_is_not_a_live_observation(tmp_path):
    path = tmp_path / 'current.fts'
    header = fits.Header({'DATE_OBS': '2023-07-11T18:07:07', 'INSTRUME': 'AIA_1', 'WAVELNTH': 193})
    fits.writeto(path, np.ones((1024, 1024), dtype=np.float32), header)
    with pytest.raises(ValueError, match='Stale'):
        read_native_fits(path, 'aia193', slot=SLOT, now=SLOT + timedelta(hours=1))


def test_nrt_fits_keeps_resolution_and_is_not_labelled_encoder_ready(tmp_path):
    path = tmp_path / 'aia.fits'
    header = fits.Header({'DATE-OBS': SLOT.isoformat().replace('+00:00', ''),
                          'INSTRUME': 'AIA_1', 'WAVELNTH': 193})
    fits.writeto(path, np.ones((1024, 1024), dtype=np.float32), header)
    data, metadata = read_native_fits(path, 'aia193', slot=SLOT, now=SLOT + timedelta(hours=1))
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
