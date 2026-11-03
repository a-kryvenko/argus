"""One-time migration of the retired local Clio archives into S3."""
import argparse
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
from types import SimpleNamespace
from uuid import UUID

import numpy as np
from PIL import Image
from sqlalchemy import select, update

from clio.db.models.file_observation import FileObservation
from clio.observations import archive, files, file_storage, sdo, sdo_browse
from pathlib import Path
import os
import gzip

def archive_root(kind):
    configured = os.getenv(f'ARGUS_{kind.upper()}_ARCHIVE')
    if configured is not None:
        return Path(configured)
    from common.config import get_config
    return (LEGACY_ROOT or get_config().data_root / 'observations') / kind


def archive_path(kind, relative):
    root = archive_root(kind).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Original path is outside its archive')
    return path


def read_gzip(path):
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(archive.MAX_ORIGINAL_BYTES + 1)
    if len(raw) > archive.MAX_ORIGINAL_BYTES:
        raise ValueError('Observation original exceeds the size limit')
    return raw


def read_receipt(row):
    if row.details:
        return dict(row.details)
    path = archive_path(row.kind, row.raw_path)
    return json.loads(path.with_suffix(path.suffix + '.receipt.json').read_text())


LEGACY_ROOT = None

KINDS = ('sdo', 'sdo_browse', 'gong', 'goes', 'donki', 'enlil')


def receipts(kind):
    """Page without keeping a transaction open during object uploads."""
    after = None
    while True:
        statement = select(FileObservation.__table__).where(FileObservation.kind == kind,
            FileObservation.error.is_(None), FileObservation.raw_path.is_not(None))
        if after is not None:
            statement = statement.where(FileObservation.id > after)
        with files.engine().connect() as connection:
            batch = connection.execute(statement.order_by(FileObservation.id).limit(200)).mappings().all()
        if not batch:
            return
        for row in batch:
            yield SimpleNamespace(**row)
        after = batch[-1]['id']


def candidates(kind):
    if kind == 'sdo':
        for path in sorted(archive_root('sdo').glob('*/*.npz')):
            yield str(path), path
    elif kind == 'sdo_browse':
        for path in sorted(archive_root('sdo_browse').glob('*.json')):
            if path.name not in ('latest.json',):
                yield str(path), path
    else:
        for row in receipts(kind):
            yield str(row.id), row


def migrate_receipt(kind, row, apply):
    metadata = read_receipt(row)
    raw = read_gzip(archive_path(kind, row.raw_path))
    if hashlib.sha256(raw).hexdigest() != row.sha256:
        raise ValueError('Local original checksum differs from the receipt')
    if apply:
        if kind in ('gong', 'goes'):
            timestamp = datetime.fromisoformat(metadata['slot_at'])
            identity = None
        else:
            timestamp, identity = row.received_at, row.id
            if timestamp.tzinfo is None:
                from datetime import UTC
                timestamp = timestamp.replace(tzinfo=UTC)
        if kind == 'donki':
            from clio.providers.donki import decode
            metadata['records'] = decode(raw)
        archive.publish_original(kind, timestamp, row.source_product, raw, metadata, id=identity)
        reference = files.find(id=UUID(metadata['file_id']))[0]
        if file_storage.read(reference) != raw:
            raise ValueError('Stored object differs from the local original')
        # Publish metadata only after the remote original has been verified.
        with files.engine().begin() as connection:
            connection.execute(update(FileObservation).where(FileObservation.id == row.id)
                .values(**archive.file_fields(kind, metadata)))
    return len(raw), 0


def migrate_scientific(path, apply):
    with np.load(path, allow_pickle=False) as source:
        image, metadata = source['image'], json.loads(str(source['metadata']))
    sdo.validate(image, metadata)
    slot, channel = sdo.utc(metadata['slot_at']), metadata['channel']
    root = archive_root('sdo')
    if path != root / f'{slot:%Y%m%dT%H0000Z}' / f'{channel}.npz':
        raise ValueError('SDO receipt differs from the archive path')
    native = path.with_suffix('.fits')
    if native.exists() and hashlib.sha256(native.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Local FITS differs from the scientific receipt')
    if apply:
        sdo.publish_scientific(image, metadata, native.read_bytes() if native.exists() else None, npz=path.read_bytes())
        for entity in ('npz.512', 'fits.512', *(['fits.original'] if native.exists() else [])):
            content = file_storage.read(sdo.file_observations.find(kind='sdo', variable=channel, entity=entity, start=slot, end=slot + timedelta(minutes=1))[0]['file'])
            if entity == 'npz.512':
                with np.load(io.BytesIO(content), allow_pickle=False) as stored:
                    if not np.array_equal(stored['image'], image, equal_nan=True) or json.loads(str(stored['metadata'])) != metadata:
                        raise ValueError('Stored NPZ differs from the local observation')
            elif entity == 'fits.original' and hashlib.sha256(content).hexdigest() != metadata['sha256']:
                raise ValueError('Stored FITS differs from the scientific receipt')
    return path.stat().st_size + (native.stat().st_size if native.exists() else 0), int(not native.exists())


def migrate_browse(path, apply):
    image = json.loads(path.read_text())
    from common.schemas.sdo_browse import SDOBrowseImage
    image = SDOBrowseImage.model_validate(image).model_dump(mode='json', exclude={'urls'})
    original = path.parent / (image['asset_id'] + '_4096.jpg')
    raw = original.read_bytes()
    digest = hashlib.sha256(b'sdo-browse-v1\0' + raw).hexdigest()
    if not image['asset_id'].endswith('_' + digest):
        raise ValueError('Browse original differs from its receipt')
    with Image.open(io.BytesIO(raw)) as source:
        if source.format != 'JPEG' or source.size != (4096, 4096):
            raise ValueError('Expected a 4096x4096 browse JPEG')
        if apply:
            timestamp = datetime.fromisoformat(image['captured_at'])
            variable = 'aia' + str(int(image['channel']))
            key = sdo.object_key(variable, timestamp, 'jpg.original', public=True)
            existing = files.find(bucket=file_storage.PUBLIC_BUCKET, key=key)
            if existing and file_storage.read(existing[0]) != raw:
                raise ValueError('Browse key is already occupied by another immutable original')
            sdo_browse.publish_image(raw, {**image, 'checked_hour': timestamp.strftime('%Y%m%d%H')})
            for entity in sdo_browse.public_entities(sdo_browse.SIZES):
                rows = sdo.file_observations.find(kind='sdo_browse', variable=variable, entity=entity,
                    start=timestamp, end=timestamp + timedelta(microseconds=1))
                stored = file_storage.read(rows[0]['file'])
                if entity == 'jpg.original' and stored != raw:
                    raise ValueError('Stored browse original differs from its receipt')
                if entity != 'jpg.original':
                    size = int(entity.split('.')[1])
                    with Image.open(io.BytesIO(stored)) as decoded:
                        if decoded.size != (size, size):
                            raise ValueError('Stored browse derivative has an invalid shape')
    return len(raw), 0


def run(*, kinds=KINDS, apply=False, limit=None, now=None):
    slots = set(sdo.hourly_slots(now or datetime.now(UTC)))
    report = dict(mode='apply' if apply else 'dry-run', sources={}, errors=[])
    for kind in kinds:
        counts = dict(scanned=0, completed=0, bytes=0, missing_originals=0, unreferenced_files=0, failed=0, skipped=0)
        report['sources'][kind] = counts
        if kind not in ('sdo', 'sdo_browse'):
            root = archive_root(kind)
            with files.engine().connect() as connection:
                known = set(connection.scalars(select(FileObservation.raw_path).where(
                    FileObservation.kind == kind, FileObservation.raw_path.is_not(None))))
            for path in root.rglob('*.gz'):
                if str(path.relative_to(root)) not in known:
                    counts['unreferenced_files'] += 1
                    report['errors'].append(dict(kind=kind, item=str(path), error='Local file has no observation receipt in the database'))
        for identity, item in candidates(kind):
            if limit is not None and counts['completed'] + counts['failed'] >= limit:
                break
            counts['scanned'] += 1
            try:
                if kind == 'sdo':
                    if item.stem not in sdo.COLLECTED_CHANNELS or datetime.strptime(
                            item.parent.name, '%Y%m%dT%H%M%SZ').replace(tzinfo=UTC) not in slots:
                        counts['skipped'] += 1
                        continue
                    size, missing = migrate_scientific(item, apply)
                elif kind == 'sdo_browse':
                    size, missing = migrate_browse(item, apply)
                else:
                    size, missing = migrate_receipt(kind, item, apply)
                counts['bytes'] += size
                counts['missing_originals'] += missing
                counts['completed'] += 1
            except Exception as exc:
                counts['failed'] += 1
                report['errors'].append(dict(kind=kind, item=identity, error=f'{type(exc).__name__}: {exc}'))
    return report


def main():
    global LEGACY_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Root of the retired local observations directory')
    parser.add_argument('--apply', action='store_true', help='Upload and link files; default is a read-only inventory')
    parser.add_argument('--kind', action='append', choices=KINDS)
    parser.add_argument('--limit', type=int, help='Maximum observations per source')
    args = parser.parse_args()
    LEGACY_ROOT = args.root
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    from common.config import get_config
    get_config()
    try:
        report = run(kinds=args.kind or KINDS, apply=args.apply, limit=args.limit)
        print(json.dumps(report, indent=2))
    finally:
        files.dispose()
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
