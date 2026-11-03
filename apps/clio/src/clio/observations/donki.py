"""DONKI response versions, lightweight poll receipts and bounded object retention."""
from datetime import UTC, datetime, timedelta
import hashlib
import logging
from uuid import uuid4

from sqlalchemy import select, insert, update, delete, or_, and_

from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.db.models.file_observation import FileObservation as Receipt
from clio.observations import files, file_storage
from clio.observations.archive import store_receipt, file_fields
from clio.providers.donki import ENDPOINTS, decode

logger = logging.getLogger(__name__)
MAX_FILE_DELETIONS = 1000


def lock():
    return ingestion_lock('donki', FILE_LOCKS['donki'])


def window(metadata):
    return tuple(datetime.fromisoformat(str(metadata[name])) for name in ('request_start', 'request_end'))


def record_poll(endpoint, *, raw=None, error=None, metadata, received_at=None):
    received_at = received_at or datetime.now(UTC)
    identity = uuid4()
    values = dict(id=identity, kind='donki_poll', source_product=endpoint, source_key=str(identity),
                  received_at=received_at, error=error)
    if error is None:
        digest = hashlib.sha256(raw).hexdigest()
        versions = select(Receipt.__table__).where(Receipt.kind == 'donki',
            Receipt.source_product == endpoint, Receipt.error.is_(None))
        with files.engine().connect() as connection:
            latest = connection.execute(versions.order_by(Receipt.received_at.desc(), Receipt.id.desc()).limit(1)).mappings().first()
            same = latest if latest and latest['details']['sha256'] == digest else connection.execute(versions.where(Receipt.details['sha256'].as_string() == digest)
                .order_by(Receipt.received_at, Receipt.id).limit(1)).mappings().first()
        if latest and latest['file_id'] and latest['details']['sha256'] == digest and window(latest['details']) == window(metadata):
            # Keep request timing/status without duplicating the response or its parsed records.
            values.update(file_id=latest['file_id'], entity='json.original')
        elif same and same['file_id']:
            receipt = dict(metadata, sha256=digest, records=decode(raw), received_at=received_at.isoformat(),
                           file_id=str(same['file_id']))
            values.update(kind='donki', **file_fields('donki', receipt))
        else:
            values = store_receipt('donki', endpoint, str(identity), id=identity, raw=raw,
                                   metadata=metadata, received_at=received_at)
    with files.engine().begin() as connection:
        connection.execute(insert(Receipt).values(**values))
    return values['id']


def _compact(endpoint, counts):
    previous, after, canonical = None, None, {}
    while True:
        query = select(Receipt.__table__).where(Receipt.kind == 'donki', Receipt.source_product == endpoint,
                                                Receipt.error.is_(None))
        if after:
            query = query.where(or_(Receipt.received_at > after[0],
                and_(Receipt.received_at == after[0], Receipt.id > after[1])))
        with files.engine().begin() as connection:
            rows = connection.execute(query.order_by(Receipt.received_at, Receipt.id).limit(200)).mappings().all()
            if not rows:
                return
            for row in rows:
                metadata = row['details'] or {}
                if not row['file_id'] or 'sha256' not in metadata:
                    previous = None
                    continue
                fingerprint = (metadata['sha256'], *window(metadata))
                target = canonical.setdefault(metadata['sha256'], row['file_id'])
                if target != row['file_id']:
                    connection.execute(update(Receipt).where(Receipt.file_id == row['file_id']).values(file_id=target))
                if fingerprint == previous:
                    connection.execute(update(Receipt).where(Receipt.id == row['id']).values(
                        kind='donki_poll', details=None))
                    counts['deduplicated'] += 1
                previous = fingerprint
            after = rows[-1]['received_at'], rows[-1]['id']


def cleanup(now, retention_days, *, max_files=MAX_FILE_DELETIONS):
    counts = dict(expired=0, deduplicated=0, files_deleted=0, failed=0)
    with lock() as acquired:
        if not acquired:
            return {**counts, 'busy': True}
        cutoff = now-timedelta(days=retention_days)
        with files.engine().begin() as connection:
            keep = []
            for endpoint in ENDPOINTS:
                boundary = select(Receipt.id).where(Receipt.kind.in_(('donki', 'donki_poll')),
                    Receipt.source_product == endpoint, Receipt.received_at < cutoff)
                # Preserve state and latest-attempt status at the beginning of the retained interval.
                for query in (boundary, boundary.where(Receipt.kind == 'donki', Receipt.error.is_(None)),
                              boundary.where(Receipt.error.is_(None))):
                    identity = connection.scalar(query.order_by(Receipt.received_at.desc(), Receipt.id.desc()).limit(1))
                    if identity:
                        keep.append(identity)
            counts['expired'] = connection.execute(delete(Receipt).where(
                Receipt.kind.in_(('donki', 'donki_poll')), Receipt.received_at < cutoff,
                Receipt.id.not_in(keep))).rowcount
        for endpoint in ENDPOINTS:
            _compact(endpoint, counts)
        from clio.observations.retention import remove_unreferenced
        counts.update(remove_unreferenced(file_storage.PRIVATE_BUCKET, 'observations/DONKI/', now=now, max_files=max_files))
        logger.info('DONKI cleanup: %s', counts)
    return counts
