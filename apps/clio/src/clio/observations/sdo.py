"""SDO observation metadata and in-memory file preparation."""
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json

import numpy as np

from clio.observations import file_storage, file_observations

OBSERVED_CHANNELS = ('aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304',
            'aia335', 'aia1600', 'hmi_m')
COLLECTED_CHANNELS = ('aia171', 'aia211')
IMAGE_SHAPE = (512, 512)
RETENTION_HOURS = 96
VERSION = 1


def utc(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('Expected timezone-aware timestamp')
    return value.astimezone(UTC)


def hourly_slots(now):
    """Scientific collection window: 96 hourly slots, newest first."""
    current = utc(now).replace(minute=0, second=0, microsecond=0)
    return [current - timedelta(hours=i) for i in range(RETENTION_HOURS)]


def validate_metadata(metadata):
    if metadata['version'] != VERSION or metadata['channel'] not in OBSERVED_CHANNELS:
        raise ValueError('Unsupported SDO observation schema/channel')
    slot = utc(metadata['slot_at'])
    observed, available = utc(metadata['observed_at']), utc(metadata['available_at'])
    if slot.minute or slot.second or slot.microsecond:
        raise ValueError('Expected whole UTC hour')
    if abs((observed - slot).total_seconds()) > 600 or observed > available:
        raise ValueError('Observation time does not match slot/availability')
    for field in ('source', 'preprocessing', 'units', 'sha256'):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f'Missing observation {field}')


def validate(image, metadata):
    validate_metadata(metadata)
    if image.shape != IMAGE_SHAPE or image.dtype != np.float32 or np.isinf(image).any() or not np.isfinite(image).any():
        raise ValueError('Expected float32 512x512 observation with finite measurements and no infinities')



def object_key(variable, timestamp, entity, *, public=False):
    timestamp = utc(timestamp)
    format, size = entity.split('.', 1)
    if public:
        if not variable.startswith('aia') or not variable[3:].isdigit():
            raise ValueError('Expected an AIA channel')
        prefix, channel = 'images/AIA', str(int(variable[3:]))
    else:
        if variable not in OBSERVED_CHANNELS:
            raise ValueError('Unknown SDO channel')
        prefix, channel = 'images/SDO', variable
    if not format.isalnum() or not size.isalnum():
        raise ValueError('Invalid file entity')
    return f'{prefix}/{channel}/{timestamp:%Y%m%d%H%M}/{size}.{format}'


def publish(content, variable, timestamp, entity, *, details, public=False, content_type):
    bucket = file_storage.PUBLIC_BUCKET if public else file_storage.PRIVATE_BUCKET
    reference = file_storage.publish(content, bucket, object_key(variable, timestamp, entity, public=public),
                                     content_type=content_type)
    product = 'nasa.aia_browse' if public else ('aia.nrt_' + variable[3:] if variable.startswith('aia') else 'hmi.nrt_m')
    return file_observations.register(variable=variable, timestamp=timestamp, entity=entity,
        bucket=bucket, key=reference['key'], uploaded_at=reference['timestamp'],
        kind='sdo_browse' if public else 'sdo', source_product=product, details=details)


def fits_bytes(image, metadata):
    from astropy.io import fits
    header = fits.Header()
    for name, value in metadata.get('header', {}).items():
        if name in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'CHECKSUM', 'DATASUM'):
            continue
        if name in ('COMMENT', 'HISTORY'):
            entries = value if isinstance(value, (list, tuple)) else [value]
            for entry in entries:
                # Older immutable NPZ receipts contain newline-joined commentary.
                for line in str(entry).splitlines() or ['']:
                    header[name] = line
        elif value is None or isinstance(value, (str, int, float, bool)):
            header[name] = value
    output = io.BytesIO()
    fits.PrimaryHDU(image, header=header).writeto(output, output_verify='silentfix')
    return output.getvalue()


def publish_scientific(image, metadata, original=None, *, npz=None):
    validate(image, metadata)
    slot, channel = utc(metadata['slot_at']), metadata['channel']
    if original is not None and hashlib.sha256(original).hexdigest() != metadata['sha256']:
        raise ValueError('Original differs from the scientific receipt')
    if npz is None:
        output = io.BytesIO()
        np.savez_compressed(output, image=image, metadata=json.dumps(metadata, allow_nan=False))
        npz = output.getvalue()
    reference, created = file_storage.upload(npz, file_storage.PRIVATE_BUCKET, object_key(channel, slot, 'npz.512'),
        content_type='application/octet-stream')
    if not created:
        # Resume an interrupted publication using the first NPZ in memory, never on local disk.
        with np.load(io.BytesIO(file_storage.read(reference)), allow_pickle=False) as stored:
            image, metadata = stored['image'], json.loads(str(stored['metadata']))
        validate(image, metadata)
        if original is not None and hashlib.sha256(original).hexdigest() != metadata['sha256']:
            raise ValueError('Original differs from the first scientific observation')
    file_observations.register(variable=channel, timestamp=slot, entity='npz.512',
        bucket=reference['bucket'], key=reference['key'], uploaded_at=reference['timestamp'],
        kind='sdo', source_product='aia.nrt_' + channel[3:] if channel.startswith('aia') else 'hmi.nrt_m', details=metadata)
    publish(fits_bytes(image, metadata), channel, slot, 'fits.512', details=metadata, content_type='application/fits')
    if original is not None:
        publish(original, channel, slot, 'fits.original', details=metadata, content_type='application/fits')
    return reference


def scientific_complete(slot, channel):
    rows = file_observations.find(kind='sdo', variable=channel, start=slot, end=slot + timedelta(minutes=1))
    return {'npz.512', 'fits.512', 'fits.original'} <= {row['entity'] for row in rows}
