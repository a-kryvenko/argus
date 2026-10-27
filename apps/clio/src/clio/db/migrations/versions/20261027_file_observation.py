"""Unify file receipts and move compressed originals out of PostgreSQL."""
from datetime import UTC
import gzip
import io
import hashlib
from pathlib import Path
from uuid import uuid4

from alembic import op
from alembic.util import CommandError
import sqlalchemy as sa

revision = '20261027_file_observation'
down_revision = '20261014_enlil_profile'
branch_labels = None
depends_on = None


def _rows(bind, table, keys):
    last = None
    columns = ','.join(keys)
    while True:
        where = '' if last is None else 'WHERE ('+columns+') > ('+','.join(':'+k for k in keys)+')'
        rows = bind.execute(sa.text(f'SELECT * FROM {table} {where} ORDER BY {columns} LIMIT 32'), last or {}).mappings().all()
        if not rows:
            return
        yield from rows
        last = {key: rows[-1][key] for key in keys}


def _original(row, kind):
    from clio.observations.files import archive_path
    if row.get('raw_path'):
        path = archive_path(kind, row['raw_path'])
        if path.is_file():
            return path.read_bytes()
    compressed = row.get('fits_gzip') if kind == 'gong' else row.get('raw_gzip')
    if compressed is None:
        return None
    from clio.observations.files import MAX_ORIGINAL_BYTES
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(MAX_ORIGINAL_BYTES + 1)
    if len(raw) > MAX_ORIGINAL_BYTES:
        raise ValueError('Legacy original exceeds the size limit')
    return raw


def upgrade():
    if op.get_context().as_sql:
        raise CommandError('File observation migration requires an online database connection '
                           'to export originals to the filesystem; --sql is not supported')
    from clio.observations.files import observation_values, store_original, store_receipt
    op.create_table('file_observation',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('kind', sa.String(32), nullable=False),
        sa.Column('source_product', sa.String(80), nullable=False),
        sa.Column('source_key', sa.Text(), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('raw_path', sa.Text()),
        sa.Column('sha256', sa.String(64)),
        sa.Column('error', sa.Text()),
        sa.CheckConstraint('(raw_path IS NULL) = (sha256 IS NULL)', name='file_observation_file_checksum'),
        sa.CheckConstraint('error IS NOT NULL OR raw_path IS NOT NULL', name='file_observation_success_file'))
    op.create_index('ix_file_observation_kind_received', 'file_observation', ['kind', 'received_at'])
    op.create_index('ix_file_observation_kind_sha256', 'file_observation', ['kind', 'sha256'])
    op.create_index('uq_file_observation_source', 'file_observation', ['kind', 'source_key'],
                    unique=True, postgresql_where=sa.text('error IS NULL'))
    bind = op.get_bind()
    target = sa.table('file_observation', *[sa.column(name) for name in (
        'id', 'kind', 'source_product', 'source_key', 'received_at', 'raw_path', 'sha256', 'error')])
    for kind, table, keys in (
        ('gong', 'gong_snapshot', ('slot_at',)),
        ('goes', 'goes_snapshot', ('slot_at', 'source_product')),
        ('donki', 'donki_poll', ('id',)),
        ('enlil', 'enlil_profile_receipt', ('id',)),
    ):
        for row in _rows(bind, table, keys):
            raw = _original(row, kind)
            if raw is not None and hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise RuntimeError(f'{table}: original checksum differs from receipt')
            if raw is None and not row.get('error'):
                raise RuntimeError(f'{table}: source original is missing; restore it before migrating')
            if kind in ('gong', 'goes'):
                metadata = {key: value for key, value in row.items() if key not in ('fits_gzip', 'raw_path')}
                record = store_original(kind, row['slot_at'], row['observed_at'], row['source_product'],
                    lambda: raw, expected=metadata,
                    relative=Path('migrated') / f'{row["slot_at"].astimezone(UTC):%Y/%m/%d}/'
                        f'{row["slot_at"].astimezone(UTC):%H%M%S}-{row["source_product"]}.gz')
                values = dict(id=uuid4(), error=None, **observation_values(kind, record))
            else:
                metadata = {key: value for key, value in row.items() if key not in ('id', 'raw_gzip', 'sha256', 'error')}
                key = str(row['id']) if kind == 'donki' else row['source_key'] or str(row['id'])
                if kind == 'enlil' and row['source_modified_at']:
                    key += '@' + row['source_modified_at'].isoformat()
                if kind == 'enlil' and row['error'] is None and bind.execute(sa.text(
                    "SELECT 1 FROM file_observation WHERE kind='enlil' AND source_key=:key AND error IS NULL"), {'key': key}).first():
                    key += '#' + str(row['id'])
                values = store_receipt(kind, row['endpoint'] if kind == 'donki' else 'noaa.enlil', key,
                    id=row['id'], raw=raw, metadata=metadata, error=row['error'], received_at=row['received_at'])
            bind.execute(target.insert().values(**values))
        op.drop_table(table)


def downgrade():
    raise RuntimeError('File observation storage requires an explicit reverse data migration')
