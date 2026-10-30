"""Hourly AIA display snapshots and their immutable JPEG/WebP derivatives."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import logging
import re
from time import monotonic

from PIL import Image
import requests

from common.schemas.sdo_browse import CHANNELS, SIZES, IMAGE_PATTERN, ASSET_PATTERN, SDOBrowseCatalog
from clio.observations.archive import archive_root, atomic_write, file_lock

logger = logging.getLogger(__name__)
MAX_BYTES = 32 * 1024 * 1024


def root():
    return archive_root('sdo_browse')


def read_latest(directory):
    try:
        return json.loads((directory / 'latest.json').read_text())
    except FileNotFoundError:
        return {}


def catalog():
    latest = read_latest(root())
    return SDOBrowseCatalog(images=[latest[c]['image'] for c in CHANNELS if c in latest])


def image_path(name):
    if not re.fullmatch(IMAGE_PATTERN, name):
        raise FileNotFoundError('Unknown SDO browse image')
    path = root() / name
    if not path.is_file():
        raise FileNotFoundError('SDO browse image is unavailable')
    return path


def download(channel):
    url = f'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_4096_{int(channel):04d}.jpg'
    started = monotonic()
    with requests.get(url, stream=True, timeout=(10, 45)) as response:
        response.raise_for_status()
        blocks, length = [], 0
        for block in response.iter_content(1024 * 1024):
            length += len(block)
            if length > MAX_BYTES or monotonic()-started > 90:
                raise ValueError('SDO browse download exceeds size/time limit')
            blocks.append(block)
        updated = response.headers.get('last-modified')
        try:
            updated = parsedate_to_datetime(updated).astimezone(UTC) if updated else None
        except (TypeError, ValueError, OverflowError):
            updated = None
        return b''.join(blocks), updated


def snapshot(directory, channel, now, previous):
    hour = now.strftime('%Y%m%d%H')
    if previous and previous['checked_hour'] == hour:
        return previous
    content, updated = download(channel)
    digest = hashlib.sha256(b'sdo-browse-v1\0'+content).hexdigest()
    if previous and previous['image']['asset_id'].endswith('_'+digest):
        return {**previous, 'checked_hour': hour}
    # Last-Modified is publication metadata, not an observation timestamp.
    if previous and updated and previous['image'].get('source_updated_at'):
        if updated <= datetime.fromisoformat(previous['image']['source_updated_at']):
            return {**previous, 'checked_hour': hour}
    with Image.open(io.BytesIO(content)) as source:
        if source.format != 'JPEG' or source.size != (4096, 4096):
            raise ValueError('Expected NASA 4096x4096 JPEG')
        original = source.convert('RGB')
    asset = f'{hour}_{channel}_{digest}'
    for size in SIZES:
        resized = original if size == 4096 else original.resize((size, size), Image.Resampling.LANCZOS)
        for extension, format in (('webp', 'WEBP'), ('jpg', 'JPEG')):
            if size == 4096 and extension == 'jpg':
                encoded = content
            else:
                buffer = io.BytesIO()
                resized.save(buffer, format=format, quality=85 if size <= 512 else 90)
                encoded = buffer.getvalue()
            atomic_write(directory / f'{asset}_{size}.{extension}', encoded)
    image = dict(channel=channel, asset_id=asset, captured_at=now.isoformat(),
                 source_updated_at=updated.isoformat() if updated else None)
    atomic_write(directory / f'{asset}.json', json.dumps(image).encode())
    return {'image': image, 'checked_hour': hour}


def collect(now):
    directory = root()
    with file_lock(directory / 'collection'):
        latest = read_latest(directory)
        failed = []
        def fetch(channel):
            try:
                return channel, snapshot(directory, channel, now, latest.get(channel))
            except (requests.RequestException, OSError, ValueError):
                logger.exception('SDO browse collection failed for AIA %s', channel)
                return channel, None
        with ThreadPoolExecutor(max_workers=3) as pool:
            for channel, record in pool.map(fetch, CHANNELS):
                if record is None:
                    failed.append(channel)
                else:
                    latest[channel] = record
        atomic_write(directory / 'latest.json', json.dumps(latest).encode())
        _cleanup(directory, latest)
        return {'status': 'partial' if failed else 'complete', 'failed': failed}


def _cleanup(directory, latest):
    records = {}
    for receipt in directory.glob('*.json'):
        if not re.fullmatch(ASSET_PATTERN, receipt.stem):
            continue
        image = json.loads(receipt.read_text())
        records.setdefault(image['channel'], []).append((image['captured_at'], receipt))
    keep = {record['image']['asset_id'] for record in latest.values()}
    for channel, images in records.items():
        current = latest.get(channel, {}).get('image', {}).get('asset_id')
        # Always retain the published image, including after an interrupted publish.
        previous = sorted((record for record in images if record[1].stem != current), reverse=True)
        keep.update(receipt.stem for _, receipt in previous[:1 if current else 2])
    removed = set()
    for path in directory.iterdir():
        asset = path.stem if re.fullmatch(ASSET_PATTERN, path.stem) else (
            path.stem.rsplit('_', 1)[0] if re.fullmatch(IMAGE_PATTERN, path.name) else None)
        if asset and asset not in keep:
            path.unlink(missing_ok=True)
            removed.add(asset)
    return len(removed)


def cleanup():
    directory = root()
    with file_lock(directory / 'collection'):
        return _cleanup(directory, read_latest(directory))
