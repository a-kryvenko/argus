"""Rebuildable, checksum-verified originals and atomic derived-cache writes."""
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from datetime import UTC, datetime, timedelta

from common.config import get_config

logger = logging.getLogger(__name__)
MAX_ORIGINAL_BYTES = 128 * 1024 * 1024


def cache_root():
    config = get_config()
    return Path(os.getenv('PROPHET_FEATURE_CACHE', str(config.data_root / 'prophet/source-cache')))


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def original(client, url, token, reference):
    path = cache_root() / reference.kind / (reference.sha256 + ('.json' if reference.kind == 'goes' else '.fits'))
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256:
            return path
    # Build the path locally; owner responses cannot redirect credentials.
    endpoint = f'{url.rstrip("/")}/internal/v1/observations/files/{reference.kind}/{reference.sha256}'
    content = bytearray()
    params = {'slot_at': reference.slot_at.isoformat()} if reference.kind == 'aia' else {}
    with client.stream('GET', endpoint, params=params, headers={'Authorization': f'Bearer {token}'}) as response:
        response.raise_for_status()
        for block in response.iter_bytes():
            content.extend(block)
            if len(content) > MAX_ORIGINAL_BYTES:
                raise ValueError('Observation original exceeds supported size')
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise ValueError('Observation checksum differs from its receipt')
    atomic_write(path, content)
    return path


def prune_source_cache():
    """Discard old derived/download caches; Clio owns the source history."""
    cutoff = datetime.now(UTC).timestamp() - timedelta(days=45).total_seconds()
    root = cache_root()
    for kind in ('aia', 'gong', 'goes'):
        for path in (root / kind).glob('*'):
            try:
                if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning('Cannot prune source cache: %s', path, exc_info=True)


