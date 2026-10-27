"""Expand file observation storage; legacy tables remain authoritative."""
from alembic import op
import sqlalchemy as sa

revision = '20261027_file_observation'
down_revision = '20261014_enlil_profile'
branch_labels = None
depends_on = None
online_compatible = True


def upgrade():
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


def downgrade():
    raise RuntimeError('Do not remove file observations while a deployed release may use them')
