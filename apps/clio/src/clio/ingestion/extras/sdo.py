"""Collect, validate and publish scientific SDO observations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import logging
import hashlib
import io
import json

import numpy as np
import requests
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from common.data import datamanager
from common.schemas.sdo_images import OBSERVED_CHANNELS
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.ingestion.utils import file_observations

IMAGE_SHAPE = (512, 512)
VERSION = 1

logger = logging.getLogger(__name__)


def utc(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('Expected timezone-aware timestamp')
    return value.astimezone(UTC)


def observation_slots(now, *, days, resolution_minutes):
    step = timedelta(minutes=resolution_minutes)
    current = utc(now)
    slot = datetime.fromtimestamp((int(current.timestamp()) // int(step.total_seconds())) * int(step.total_seconds()), UTC)
    lower = current - timedelta(days=days)
    slots = []
    while slot >= lower:
        slots.append(slot)
        slot -= step
    return slots


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


def publish(content, variable, timestamp, entity, *, details, public=False, content_type, source=None):
    bucket = 'argus-public' if public else 'argus-private'
    key = object_key(variable, timestamp, entity, public=public)
    uploaded, _ = datamanager.create_immutable_file(content, bucket, key, content_type=content_type)
    product = 'nasa.aia_browse' if public else ('aia.nrt_' + variable[3:] if variable.startswith('aia') else 'hmi.nrt_m')
    return file_observations.register(variable=variable, timestamp=timestamp, entity=entity,
        bucket=bucket, key=key, uploaded_at=uploaded,
        kind='sdo_browse' if public else 'sdo', source_product=source or product, details=details)


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


def publish_scientific(image, metadata, original=None, *, npz=None, source=None):
    validate(image, metadata)
    slot, channel = utc(metadata['slot_at']), metadata['channel']
    if original is not None and hashlib.sha256(original).hexdigest() != metadata['sha256']:
        raise ValueError('Original differs from the scientific receipt')
    if npz is None:
        output = io.BytesIO()
        np.savez_compressed(output, image=image, metadata=json.dumps(metadata, allow_nan=False))
        npz = output.getvalue()
    key = object_key(channel, slot, 'npz.512')
    uploaded, created = datamanager.create_immutable_file(npz, 'argus-private', key,
        content_type='application/octet-stream')
    if not created:
        # Resume an interrupted publication using the first NPZ in memory, never on local disk.
        with np.load(io.BytesIO(datamanager.read_file('argus-private', key, max_bytes=128 * 1024 * 1024)), allow_pickle=False) as stored:
            image, metadata = stored['image'], json.loads(str(stored['metadata']))
        validate(image, metadata)
        if original is not None and hashlib.sha256(original).hexdigest() != metadata['sha256']:
            raise ValueError('Original differs from the first scientific observation')
    reference = file_observations.register(variable=channel, timestamp=slot, entity='npz.512',
        bucket='argus-private', key=key, uploaded_at=uploaded,
        kind='sdo', source_product=source or ('aia.nrt_' + channel[3:] if channel.startswith('aia') else 'hmi.nrt_m'), details=metadata)
    publish(fits_bytes(image, metadata), channel, slot, 'fits.512', details=metadata, content_type='application/fits', source=source)
    if original is not None:
        publish(original, channel, slot, 'fits.original', details=metadata, content_type='application/fits', source=source)
    return reference


def scientific_complete(slot, channel):
    rows = file_observations.find(kind='sdo', variable=channel, slot_at=slot)
    return {'npz.512', 'fits.512', 'fits.original'} <= {row['entity'] for row in rows}


def collect_one(slot, channel, *, clock, source=None):
    try:
        now = clock()
        if channel not in OBSERVED_CHANNELS:
            return 'unavailable'
        if scientific_complete(slot, channel):
            return 'existing'
        from clio.providers import sdo as provider
        image, metadata, original = provider.load_observation(channel, slot, now=now)
        now = clock()
        metadata = dict(metadata, version=VERSION, slot_at=slot.isoformat(), channel=channel,
                        available_at=now.isoformat())
        publish_scientific(image, metadata, original, source=source)
        return 'saved'
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return 'unavailable'
        logger.warning('SDO download failed: %s %s: %s', slot, channel, exc)
    except (requests.RequestException, ValueError, OSError, BotoCoreError, ClientError, SQLAlchemyError) as exc:
        logger.warning('SDO observation failed: %s %s: %s', slot, channel, exc)
    return 'failed'


def collect(policy, *, channel, source, mode, now):
    lane = 'sdo-live' if mode == 'live' else 'sdo-warmup'
    with ingestion_lock(lane, FILE_LOCKS[lane]) as acquired:
        if not acquired:
            return {'status': 'busy', 'failed': 0}
        resolution = policy.live_resolution_minutes if mode == 'live' else policy.resolution_minutes
        slots = observation_slots(now, days=policy.retention_days, resolution_minutes=resolution)
        horizon = timedelta(days=policy.backfill_days) if mode == 'backfill' else timedelta(minutes=policy.freshness_minutes)
        slots = [slot for slot in slots if slot >= now - horizon]
        report = dict(saved=0, existing=0, unavailable=0, failed=0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for state in pool.map(lambda slot: collect_one(slot, channel, clock=lambda: datetime.now(UTC), source=source), slots):
                report[state] += 1
        report['status'] = 'partial' if report['failed'] or (mode == 'backfill' and report['unavailable']) else 'complete'
        return report
