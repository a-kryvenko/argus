"""Hourly original AIA193 snapshots with actual receipt provenance."""
from alembic import op
import sqlalchemy as sa
revision='20260923_aia_snapshot'
down_revision='20260917_measurement_receipt'
branch_labels=None
depends_on=None


def upgrade():
    op.create_table('aia_snapshot',sa.Column('slot_at',sa.DateTime(timezone=True),primary_key=True),
        sa.Column('observed_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('available_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('sha256',sa.String(64),nullable=False),sa.Column('raw_path',sa.Text,nullable=False),sa.Column('cache_path',sa.Text,nullable=False),
        sa.Column('b0_deg',sa.Float,nullable=False),sa.Column('valid_fraction',sa.Float,nullable=False),sa.Column('carrington_lon',sa.Float,nullable=False))
    op.create_index('ix_aia_snapshot_observed_at','aia_snapshot',['observed_at'])
    op.create_index('ix_aia_snapshot_available_at','aia_snapshot',['available_at'])


def downgrade():
    op.drop_table('aia_snapshot')
