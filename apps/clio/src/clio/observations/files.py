"""Immutable source files and first-receipt metadata, without model processing."""
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile


def archive_root(kind):
    from common.config import get_config
    return Path(os.getenv(f'ARGUS_{kind.upper()}_ARCHIVE', str(get_config().data_root / 'observations' / kind)))


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
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def store_original(kind, slot, observed, source, download, *, expected=None, root=None, relative=None, metadata=None):
    """Recover files only if their checksum agrees with the first receipt."""
    root = root or archive_root(kind)
    relative = relative or raw_path(kind, slot)
    path = root / relative
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
        content = path.read_bytes() if path.exists() else download()
        if content is None:
            return None
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
            atomic_write(path, content)
        if not receipt.exists():
            serializable = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in record.items()}
            atomic_write(receipt, json.dumps(serializable).encode())
        return {**record, **{k: datetime.fromisoformat(str(record[k])) for k in ('slot_at', 'observed_at', 'available_at')}}
