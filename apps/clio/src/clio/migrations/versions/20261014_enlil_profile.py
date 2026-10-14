"""Version compact NOAA ENLIL profiles separately from NASA DONKI."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20261014_enlil_profile'
down_revision = '20261013_donki_poll'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('enlil_profile_receipt',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=False),
        sa.Column('source_key', sa.Text()),
        sa.Column('source_modified_at', sa.DateTime(timezone=True)),
        sa.Column('model_created_at', sa.DateTime(timezone=True)),
        sa.Column('sha256', sa.String(64)),
        sa.Column('raw_gzip', sa.LargeBinary()),
        sa.Column('payload', postgresql.JSONB()),
        sa.Column('error', sa.Text()))
    op.create_index('ix_enlil_profile_received', 'enlil_profile_receipt', ['received_at'])


def downgrade():
    op.drop_table('enlil_profile_receipt')
