"""Require a complete file archive before switching off legacy reads and writes."""
from datetime import datetime
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path

from alembic import op
from alembic.util import CommandError
import sqlalchemy as sa

revision = '20261027_file_cutover'
down_revision = '20261027_file_observation'
branch_labels = None
depends_on = None
online_compatible = True

SOURCES = {'gong': 'gong_snapshot', 'goes': 'goes_snapshot',
           'donki': 'donki_poll', 'enlil': 'enlil_profile_receipt'}
MAX_BYTES = 128 * 1024 * 1024


def _verify_file(row):
    kind = row['kind']
    if kind not in SOURCES:
        raise ValueError('Unknown observation kind')
    configured = os.getenv(f'ARGUS_{kind.upper()}_ARCHIVE')
    if configured is None:
        from common.config import get_config
        configured = get_config().data_root / 'observations' / kind
    root = Path(configured).resolve()
    path = (root / row['raw_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Original is outside its archive')
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('Original exceeds the size limit')
            digest.update(chunk)
    if digest.hexdigest() != row['sha256']:
        raise ValueError('Original checksum differs from receipt')
    metadata = json.loads(path.with_suffix(path.suffix + '.receipt.json').read_text())
    if metadata['sha256'] != row['sha256']:
        raise ValueError('Sidecar checksum differs from receipt')
    time = metadata['available_at' if kind in ('gong', 'goes') else 'received_at']
    if datetime.fromisoformat(time) != row['received_at']:
        raise ValueError('Sidecar receipt time differs from database')
    if row['error'] is None:
        required = {'gong': ('slot_at', 'observed_at', 'source_url'),
                    'goes': ('slot_at', 'observed_at'),
                    'donki': ('request_start', 'request_end'),
                    'enlil': ('source_key', 'source_url', 'source_modified_at', 'model_created_at', 'payload')}
        if any(metadata.get(key) is None for key in required[kind]):
            raise ValueError('Sidecar metadata is incomplete')


def upgrade():
    if op.get_context().as_sql:
        raise CommandError('File observation cutover requires an online database connection; --sql is not supported')
    bind = op.get_bind()
    for kind, table in SOURCES.items():
        snapshot = kind in ('gong', 'goes')
        identity = ("md5('" + kind + ":' || to_char(legacy.slot_at AT TIME ZONE 'UTC', "
                    "'YYYY-MM-DD\"T\"HH24:MI:SS.US') || ':' || legacy.source_product)::uuid") if snapshot else 'legacy.id'
        source = 'legacy.source_product' if snapshot else 'legacy.endpoint' if kind == 'donki' else "'noaa.enlil'"
        missing = bind.scalar(sa.text(f'''
            SELECT count(*) FROM clio.{table} legacy
            LEFT JOIN clio.file_observation f ON f.id = {identity}
            WHERE f.id IS NULL OR f.kind <> :kind OR f.source_product <> {source}
               OR f.received_at <> legacy.{'available_at' if snapshot else 'received_at'}
               OR f.sha256 IS DISTINCT FROM legacy.sha256
               OR f.error IS DISTINCT FROM {'NULL' if snapshot else 'legacy.error'}
        '''), {'kind': kind})
        if missing:
            raise CommandError(f'File observation cutover blocked: {kind} has {missing} unarchived or conflicting receipts; '
                               'finish backfill-files --check on the stage 1 release before deploying')
    rows = bind.execute(sa.text('''
        SELECT id,kind,raw_path,sha256,received_at,error FROM clio.file_observation
        WHERE raw_path IS NOT NULL
    ''').execution_options(stream_results=True))
    checked = 0
    try:
        for row in rows.mappings():
            try:
                _verify_file(row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CommandError(f'File observation cutover blocked: {row["kind"]} {row["id"]}: {exc}') from None
            checked += 1
            if checked % 128 == 0:
                logging.getLogger(__name__).info('Verified %s file observations', checked)
    finally:
        rows.close()
    logging.getLogger(__name__).info('File observation cutover verified: %s originals', checked)


def downgrade():
    raise CommandError('Restore the stage 1 application without downgrading this checkpoint')
