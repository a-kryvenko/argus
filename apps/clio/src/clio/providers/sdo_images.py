"""Download individual scientific images for eight AIA channels and HMI LOS.

Scientific 1024-pixel NRT and 4096-pixel FITS preserve their source resolution.
Neither is labelled as calibrated or as a ready-to-use encoder input.
Calibration and construction of a registered encoder tensor are separate steps.
"""
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import requests

from common.sdo_images import OBSERVED_CHANNELS, utc

MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024


def fits_observed_at(header):
    from astropy.time import Time
    # T_OBS in HMI products is usually explicitly TAI, not UTC.
    value = header.get('T_OBS') or header.get('DATE_OBS') or header.get('DATE-OBS')
    if not value:
        raise ValueError('Missing FITS observation time')
    value = str(value).strip()
    scale = str(header.get('TIMESYS', 'UTC')).strip().lower()
    if value.endswith(('_TAI', '_UTC')):
        value, scale = value.rsplit('_', 1)
        scale = scale.lower()
        date, time = value.split('_', 1)
        value = date.replace('.', '-') + 'T' + time
    elif len(value) == 10 and header.get('TIME-OBS'):
        value += 'T' + header['TIME-OBS']
    return Time(value.rstrip('Z'), scale=scale).utc.to_datetime(timezone=UTC)


def read_native_fits(path, channel, *, slot, now, tolerance_seconds=120):
    from astropy.io import fits
    if channel not in OBSERVED_CHANNELS:
        raise ValueError('Clio accepts real observations only')
    slot, now = utc(slot), utc(now)
    if slot.minute or slot.second or slot.microsecond or tolerance_seconds < 0:
        raise ValueError('Expected whole UTC hour and nonnegative time tolerance')
    with fits.open(path, memmap=False) as hdus:
        hdu = next((h for h in hdus if h.data is not None and h.data.ndim == 2), None)
        if hdu is None:
            raise ValueError('FITS has no 2D image')
        header = hdu.header.copy()
        observed = fits_observed_at(header)
        if observed > now or abs((observed - slot).total_seconds()) > tolerance_seconds:
            raise ValueError('Stale/future FITS or observation does not match requested slot')
        data = np.asarray(hdu.data, dtype=np.float32).copy()
    if data.shape not in ((1024, 1024), (4096, 4096)):
        raise ValueError(f'Expected scientific 1024x1024 or 4096x4096 FITS; received {data.shape}')
    instrument = str(header.get('INSTRUME', '')).upper()
    if channel.startswith('aia'):
        if 'AIA' not in instrument or int(header.get('WAVELNTH', -1)) != int(channel[3:]):
            raise ValueError('FITS instrument/wavelength does not match requested AIA channel')
    elif channel.startswith('hmi'):
        units = str(header.get('BUNIT', '')).strip().lower()
        expected = {'hmi_m': {'gauss', 'g'}}
        if 'HMI' not in instrument or units not in expected[channel]:
            raise ValueError('FITS instrument/units do not match requested HMI channel')
    if not np.isfinite(data).any():
        raise ValueError('FITS has no finite measurements')
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return data, dict(channel=channel, slot_at=slot.isoformat(), observed_at=observed.isoformat(),
                      available_at=now.isoformat(), sha256=digest, native_shape=list(data.shape),
                      quality=header.get('QUALITY'), header=dict(header),
                      preprocessing='fits-unregistered-v1')


def download_observation(url, channel, *, slot, now, session=None):
    """Fetch to temporary storage and validate before returning a source image.

    HTTP auth errors are propagated; no protected endpoints are scraped or
    silently replaced with JPEG previews or archival data.
    """
    client = session or requests
    with TemporaryDirectory(prefix='clio-sdo-') as temporary:
        path = Path(temporary) / 'observation.fits'
        deadline = monotonic() + 120
        with client.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            size = 0
            with path.open('wb') as stream:
                for block in response.iter_content(1024 * 1024):
                    if monotonic() > deadline:
                        raise requests.Timeout('SDO download exceeded total time budget')
                    size += len(block)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError('Oversized SDO FITS download')
                    stream.write(block)
        image, metadata = read_native_fits(path, channel, slot=slot, now=now)
        metadata['source'] = url
        metadata['available_at'] = datetime.now(UTC).isoformat()
        return image, metadata


def observation_url(channel, slot):
    """Public numeric NRT products; HMI filenames use nominal TAI hours."""
    slot = utc(slot)
    if channel not in OBSERVED_CHANNELS or slot.minute or slot.second or slot.microsecond:
        raise ValueError('Expected observed channel and whole UTC hour')
    base = 'https://jsoc1.stanford.edu/data'
    if channel.startswith('aia'):
        return (f'{base}/aia/synoptic/nrt/{slot:%Y/%m/%d/H%H00}/'
                f'AIA{slot:%Y%m%d_%H0000}_{int(channel[3:]):04d}.fits')
    return f'{base}/hmi/fits/{slot:%Y/%m/%d}/hmi.M_720s_nrt.{slot:%Y%m%d_%H0000}_TAI.fits'


def prepare_observation(image, metadata):
    """Area means in source units, retaining NaN where no measurement exists.

    This is observation reduction, not Surya normalization or co-registration.
    The adjusted WCS describes the reduced grid, including pixel-center offsets.
    """
    factor = image.shape[0] // 512
    if image.shape not in ((1024, 1024), (4096, 4096)):
        raise ValueError('Unsupported source shape')
    blocks = image.reshape(512, factor, 512, factor)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=(1, 3))
    sums = np.where(finite, blocks, 0).sum(axis=(1, 3), dtype=np.float64)
    reduced = np.full((512, 512), np.nan, dtype=np.float32)
    np.divide(sums, counts, out=reduced, where=counts > 0)
    def scalar(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)
    source_header = {k: scalar(v) for k, v in metadata['header'].items()}
    header = dict(source_header)
    for axis in (1, 2):
        header[f'NAXIS{axis}'] = 512
        key = f'CRPIX{axis}'
        if key in header:
            header[key] = (header[key] - 0.5) / factor + 0.5
        key = f'CDELT{axis}'
        if key in header:
            header[key] *= factor
        for other in (1, 2):
            key = f'CD{axis}_{other}'
            if key in header:
                header[key] *= factor
    return reduced, dict(metadata, header=header, source_header=source_header,
                         preprocessing='sdo-area-mean-unregistered-v1',
                         units='Gauss' if metadata['channel'] == 'hmi_m' else 'DN',
                         missing_pixels='NaN; finite-only block mean; no imputation',
                         valid_fraction=float(np.isfinite(reduced).mean()))
