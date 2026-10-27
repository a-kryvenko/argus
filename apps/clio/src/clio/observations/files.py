"""Immutable source files and first-receipt metadata, without model processing."""
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import gzip
import fcntl
import hashlib
import json
import os
import tempfile
from uuid import UUID


def archive_root(kind):
    configured = os.getenv(f'ARGUS_{kind.upper()}_ARCHIVE')
    if configured is not None:
        return Path(configured)
    from common.config import get_config
    return get_config().data_root / 'observations' / kind


def raw_path(kind, slot):
    suffix = '.json' if kind == 'goes' else '.fits'
    return Path(f'{slot:%Y/%m/%d}/{kind}_{slot:%Y%m%dT%H%M%S}{suffix}')


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_write(path, content):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def store_original(kind, slot, observed, source, download, *, expected=None, root=None, relative=None, metadata=None):
    """Recover files only if their checksum agrees with the first receipt."""
    root = root or archive_root(kind)
    relative = relative or raw_path(kind, slot)
    relative = Path(str(relative) if str(relative).endswith('.gz') else str(relative) + '.gz')
    path = archive_path(kind, relative, root=root)
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    with file_lock(path):
        record = json.loads(receipt.read_text()) if receipt.exists() else expected
        if record:
            record = dict(record)
            if datetime.fromisoformat(str(record['slot_at'])) != slot:
                raise ValueError('File receipt belongs to another slot')
            if expected and any(str(record[k]) != str(expected[k]) for k in ('sha256',)):
                raise ValueError('File receipt conflicts with the stored checksum')
            if expected and any(datetime.fromisoformat(str(record[k])) != datetime.fromisoformat(str(expected[k]))
                                for k in ('observed_at', 'available_at')):
                raise ValueError('File receipt conflicts with the stored time')
        content = read_gzip(path) if path.exists() else download()
        if content is None:
            return None
        if len(content) > MAX_ORIGINAL_BYTES:
            raise ValueError('Observation original exceeds the size limit')
        digest = hashlib.sha256(content).hexdigest()
        if record and digest != record['sha256']:
            raise ValueError('Original differs from the recorded SHA256')
        if record is None:
            record = dict(slot_at=slot.isoformat(), observed_at=observed.isoformat(),
                          available_at=datetime.now(UTC).isoformat(), sha256=digest, source_product=source)
        if metadata and not receipt.exists() and expected is None:
            record.update(metadata)
        record['raw_path'] = str(relative)
        if not path.exists():
            atomic_write(path, gzip.compress(content, mtime=0))
        if not receipt.exists():
            serializable = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in record.items()}
            atomic_write(receipt, json.dumps(serializable).encode())
        return {**record, **{k: datetime.fromisoformat(str(record[k])) for k in ('slot_at', 'observed_at', 'available_at')}}


MAX_ORIGINAL_BYTES = 128 * 1024 * 1024


def archive_path(kind, relative, *, root=None):
    root = Path(root or archive_root(kind)).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Original path is outside its archive')
    return path


def read_gzip(path):
    with gzip.open(path, 'rb') as stream:
        content = stream.read(MAX_ORIGINAL_BYTES + 1)
    if len(content) > MAX_ORIGINAL_BYTES:
        raise ValueError('Observation original exceeds the size limit')
    return content


def read_original(kind, relative, sha256):
    content = read_gzip(archive_path(kind, relative))
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError('Original checksum differs from receipt')
    return content


def read_receipt(row):
    path = archive_path(row.kind, row.raw_path)
    record = json.loads(path.with_suffix(path.suffix + '.receipt.json').read_text())
    if record['sha256'] != row.sha256:
        raise ValueError('File receipt conflicts with the stored checksum')
    return record


def source_key(slot, source):
    return slot.astimezone(UTC).isoformat() + ':' + source


def observation_values(kind, record):
    # Keep snapshot IDs stable across retries and existing archived observations.
    identity = f'{kind}:{record["slot_at"].astimezone(UTC):%Y-%m-%dT%H:%M:%S.%f}:{record["source_product"]}'
    id = UUID(hashlib.md5(identity.encode(), usedforsecurity=False).hexdigest())
    return dict(id=id, kind=kind, source_product=record['source_product'],
        source_key=source_key(record['slot_at'], record['source_product']),
        received_at=record['available_at'], raw_path=record['raw_path'], sha256=record['sha256'])


def snapshot_receipt(row):
    record = read_receipt(row)
    return {**record, **{key: datetime.fromisoformat(record[key]) for key in ('slot_at', 'observed_at')},
            'available_at': row.received_at, 'source_product': row.source_product,
            'raw_path': row.raw_path, 'sha256': row.sha256}


def store_receipt(kind, source_product, key, *, raw=None, metadata=None, received_at=None, error=None, id=None):
    from uuid import uuid4
    id = id or uuid4()
    received_at = received_at or datetime.now(UTC)
    values = dict(id=id, kind=kind, source_product=source_product, source_key=key,
                  received_at=received_at, raw_path=None, sha256=None, error=error)
    if raw is None:
        if error is None:
            raise ValueError('A successful file observation requires an original')
        return values
    if len(raw) > MAX_ORIGINAL_BYTES:
        raise ValueError('Observation original exceeds the size limit')
    digest = hashlib.sha256(raw).hexdigest()
    relative = Path(f'{received_at:%Y/%m/%d}/{id}.gz')
    path = archive_path(kind, relative)
    with file_lock(path):
        if path.exists():
            read_original(kind, relative, digest)
        else:
            atomic_write(path, gzip.compress(raw, mtime=0))
        receipt = {**(metadata or {}), 'sha256': digest, 'received_at': received_at.isoformat()}
        atomic_write(path.with_suffix(path.suffix + '.receipt.json'), json.dumps(receipt, default=str).encode())
    return {**values, 'raw_path': str(relative), 'sha256': digest}
