"""AIA193 originals and recoverable derivatives with immutable first receipts."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile
from zipfile import BadZipFile

import numpy as np
import requests

from clio.domains.aia.extraction import extract_frame

BASE_URL = 'https://jsoc1.stanford.edu/data/aia/synoptic'
NRT_URL = BASE_URL + '/nrt'
SOURCE_URLS = {'aia.synoptic_193': BASE_URL, 'aia.nrt_193': NRT_URL}
MAX_BYTES = 32 * 1024 * 1024


def hourly_slots(now, history_days=40):
    if now.tzinfo is None or not 1 <= history_days <= 60:
        raise ValueError('Aware time and 1–60 history days required')
    end = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=history_days)
    return [start + timedelta(hours=h) for h in range(history_days * 24 + 1)]


def snapshot_paths(slot, root):
    if slot.tzinfo is None:
        raise ValueError('Expected aware hourly slot')
    slot = slot.astimezone(UTC)
    if slot.minute or slot.second or slot.microsecond:
        raise ValueError('Expected aware hourly slot')
    raw = Path(root) / f'{slot:%Y/%m/%d}/AIA{slot:%Y%m%d_%H%M}_0193.fits'
    return raw, raw.with_suffix('.fits.npz'), raw.with_suffix('.fits.json')


@contextmanager
def snapshot_lock(raw):
    """Serialize only one slot, including concurrent live/backfill processes."""
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.with_suffix('.fits.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _time(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('AIA receipt contains a naive timestamp')
    return value.astimezone(UTC)


def _known_record(receipt, expected, slot):
    record = json.loads(receipt.read_text()) if receipt.exists() else expected
    if record is not None:
        if _time(record['slot_at']) != slot:
            raise ValueError('AIA receipt belongs to another slot')
        if expected is not None:
            if record['sha256'] != expected['sha256'] or record.get('rejected'):
                raise ValueError('AIA receipt conflicts with stored observation')
            for field in ('observed_at', 'available_at'):
                if _time(record[field]) != _time(expected[field]):
                    raise ValueError('AIA receipt conflicts with stored availability/time')
        record = dict(record)
        for field in ('slot_at', 'observed_at', 'available_at'):
            if field in record:
                record[field] = _time(record[field]).isoformat()
    return record


def _valid_cache(cache, record):
    try:
        with np.load(cache, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            return (saved['dark'].shape == (61, 61) and saved['ratio'].shape == (61, 61)
                    and metadata['sha256'] == record['sha256']
                    and _time(metadata['observed_at']) == _time(record['observed_at']))
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def _download(slot, temporary, base_url=BASE_URL):
    stamp = slot.strftime('%Y%m%d_%H%M%S' if base_url == NRT_URL else '%Y%m%d_%H%M')
    name = f'AIA{stamp}_0193.fits'
    url = f'{base_url}/{slot:%Y/%m/%d}/H{slot:%H}00/{name}'
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        if response.status_code == 404:
            return False
        response.raise_for_status()
        size = 0
        with temporary.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                size += len(block)
                if size > MAX_BYTES:
                    raise ValueError('Oversized AIA FITS')
                stream.write(block)
    return True


def fetch_snapshot(slot, root, expected=None, *, source_product=None):
    """Recover missing artifacts; never replace an existing original or receipt.

    ``expected`` is a persisted observation receipt when the DB already knows
    this slot. A re-download must match its hash and keeps its first availability.
    """
    raw, cache, receipt = snapshot_paths(slot, root)
    slot = slot.astimezone(UTC)
    with snapshot_lock(raw):
        record = _known_record(receipt, expected, slot)
        temporary = None
        try:
            if raw.exists():
                source = raw
            else:
                fd, name = tempfile.mkstemp(prefix=raw.name + '.', suffix='.part', dir=raw.parent)
                os.close(fd)
                temporary = Path(name)
                product = (record or {}).get('source_product') or (expected or {}).get('source_product') or source_product
                downloaded = (_download(slot, temporary, SOURCE_URLS[product]) if product
                              else _download(slot, temporary))
                if not downloaded:
                    return None
                source = temporary
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if record is not None and digest != record['sha256']:
                raise ValueError('AIA original differs from the recorded SHA256')
            received = datetime.now(UTC)
            if record is not None and record.get('rejected'):
                if temporary is not None:
                    os.link(temporary, raw)
                return None
            if record is None or not _valid_cache(cache, record):
                try:
                    product = (record or {}).get('source_product') or (expected or {}).get('source_product') or source_product
                    dark, ratio, meta = (extract_frame(source, allow_nrt=True)
                                         if product == 'aia.nrt_193' else extract_frame(source))
                except ValueError as exc:
                    if record is not None:
                        # A previously accepted observation cannot become a new
                        # rejected receipt during derivative recovery.
                        raise
                    if temporary is not None:
                        os.link(temporary, raw)
                    _atomic_json(receipt, dict(rejected=True, slot_at=slot.isoformat(),
                                              available_at=received.isoformat(), sha256=digest, reason=str(exc)))
                    return None
                observed = _time(meta['observed_at'])
                if abs((observed - slot).total_seconds()) > 600 or observed > received:
                    raise ValueError('AIA timestamp does not match requested slot')
                if meta['sha256'] != digest:
                    raise ValueError('AIA extraction checksum does not match original')
                if record is not None:
                    if observed != _time(record['observed_at']):
                        raise ValueError('AIA extraction time conflicts with stored observation')
                    for field in ('b0_deg', 'valid_fraction', 'carrington_lon'):
                        if meta[field] != record[field]:
                            raise ValueError('AIA extraction metadata conflicts with stored observation')
                else:
                    record = dict(slot_at=slot.isoformat(), observed_at=observed.isoformat(),
                                  available_at=received.isoformat(), sha256=digest,
                                  b0_deg=meta['b0_deg'], valid_fraction=meta['valid_fraction'],
                                  carrington_lon=meta['carrington_lon'])
                    if source_product:
                        record['source_product'] = source_product
                if temporary is not None:
                    os.link(temporary, raw)
                    temporary.unlink()
                    temporary = None
                meta['path'] = str(raw.resolve())
                cache_temp = cache.with_suffix(cache.suffix + '.part')
                try:
                    with cache_temp.open('wb') as stream:
                        np.savez_compressed(stream, dark=dark, ratio=ratio, metadata=json.dumps(meta))
                    cache_temp.replace(cache)
                finally:
                    cache_temp.unlink(missing_ok=True)
            elif temporary is not None:
                os.link(temporary, raw)
            record = {**record, 'raw_path': str(raw.resolve()), 'cache_path': str(cache.resolve())}
            if not receipt.exists():
                _atomic_json(receipt, record)
            return record
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
