"""Retain immutable DONKI poll versions and receipt times."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20261013_donki_poll'
down_revision = '20261004_drop_aia_snapshot'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('donki_poll',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('endpoint', sa.String(32), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('request_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('request_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=False),
        sa.Column('sha256', sa.String(64)),
        sa.Column('raw_gzip', sa.LargeBinary()),
        sa.Column('payload', postgresql.JSONB()),
        sa.Column('error', sa.Text()))
    op.create_index('ix_donki_poll_endpoint_received', 'donki_poll', ['endpoint', 'received_at'])


def downgrade():
    op.drop_table('donki_poll')
