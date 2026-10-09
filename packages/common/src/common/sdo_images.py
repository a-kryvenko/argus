"""Versioned hourly SDO observation storage shared by Clio and Prophet.

Only observations belong here. Missing channels have no files; reconstructed
images must stay in the forecasting layer. Arrays retain their preprocessing
identifier so consumers cannot silently mix different numerical conventions.
"""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

OBSERVED_CHANNELS = ('aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304',
            'aia335', 'aia1600', 'hmi_m')
# PROSWIN needs native 171/211 FITS; retain 193 for existing archive consumers.
ORIGINAL_CHANNELS = ('aia171', 'aia193', 'aia211')
IMAGE_SHAPE = (512, 512)
RETENTION_HOURS = 45 * 24
VERSION = 1


def utc(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('Expected timezone-aware timestamp')
    return value.astimezone(UTC)


def hourly_slots(now):
    """Retained hourly slots, newest first (45 days including the current hour)."""
    current = utc(now).replace(minute=0, second=0, microsecond=0)
    return [current - timedelta(hours=i) for i in range(RETENTION_HOURS)]


def archive_root():
    override = os.getenv('ARGUS_SDO_ARCHIVE')
    if override is not None:
        return Path(override)
    from common.config import get_config
    return get_config().data_root / 'observations/sdo'


def image_path(root, slot, channel):
    slot = utc(slot)
    if slot.minute or slot.second or slot.microsecond or channel not in OBSERVED_CHANNELS:
        raise ValueError('Expected whole UTC hour and an observed channel')
    return Path(root) / f'{slot:%Y%m%dT%H0000Z}' / f'{channel}.npz'


def original_path(root, slot, channel):
    return image_path(root, slot, channel).with_suffix('.fits')


def save_original(root, slot, channel, content, *, now):
    """Restore a retained original only when it matches the first source receipt."""
    with archive_lock(root):
        if utc(slot) not in hourly_slots(now):
            raise ValueError('Observation is outside the retained window')
        metadata = read_metadata(root, slot, channel)
        if hashlib.sha256(content).hexdigest() != metadata['sha256']:
            raise ValueError('Original differs from the recorded SHA256')
        path = original_path(root, slot, channel)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']:
            return False
        fd, name = tempfile.mkstemp(prefix='.', suffix='.part', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
            Path(name).replace(path)
        finally:
            Path(name).unlink(missing_ok=True)
        return True


@contextmanager
def archive_lock(root):
    """One persistent lock protects publication and retention across processes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.archive.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


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


def read_metadata(root, slot, channel):
    """Read the receipt without decompressing the image array for a catalog."""
    with np.load(image_path(root, slot, channel), allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
    validate_metadata(metadata)
    if utc(metadata['slot_at']) != utc(slot) or metadata['channel'] != channel:
        raise ValueError('Observation does not match its filename')
    return metadata


def read_image(root, slot, channel, *, as_of=None):
    with np.load(image_path(root, slot, channel), allow_pickle=False) as saved:
        image = saved['image']
        metadata = json.loads(str(saved['metadata']))
    validate(image, metadata)
    if utc(metadata['slot_at']) != utc(slot) or metadata['channel'] != channel:
        raise ValueError('Observation does not match its filename')
    if as_of is not None and utc(metadata['available_at']) > utc(as_of):
        raise ValueError('Observation was not available at forecast issue time')
    return image, metadata


def save_image(root, slot, channel, image, metadata, *, now):
    """Atomically publish a first observation; refuse writes outside retention."""
    metadata = dict(metadata, version=VERSION, slot_at=utc(slot).isoformat(), channel=channel)
    image = np.asarray(image, dtype=np.float32)
    validate(image, metadata)
    if utc(metadata['available_at']) > utc(now):
        raise ValueError('Observation availability is in the future')
    with archive_lock(root):
        if utc(slot) not in hourly_slots(now):
            raise ValueError('Observation is outside the retained window')
        path = image_path(root, slot, channel)
        if path.exists():
            read_image(root, slot, channel)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.', suffix='.part', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, image=image, metadata=json.dumps(metadata))
            Path(name).replace(path)
        finally:
            Path(name).unlink(missing_ok=True)
    return True


def prune(root, *, now):
    """Remove only owned channel files in expired hourly directories."""
    cutoff = hourly_slots(now)[-1]
    removed = 0
    with archive_lock(root):
        for directory in Path(root).iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                slot = datetime.strptime(directory.name, '%Y%m%dT%H0000Z').replace(tzinfo=UTC)
            except ValueError:
                continue
            if slot >= cutoff:
                continue
            for channel in OBSERVED_CHANNELS:
                path = directory / f'{channel}.npz'
                if path.is_file():
                    path.unlink()
                    removed += 1
                original = directory / f'{channel}.fits'
                original.unlink(missing_ok=True)
            if not any(directory.iterdir()):
                directory.rmdir()
    return removed
