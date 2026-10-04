"""Store originals independently of Prophet computations; retain legacy data."""
from alembic import op
import sqlalchemy as sa

revision = '20261004_raw_observation_files'
down_revision = '20261003_unified_measurement'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('gong_snapshot', sa.Column('raw_path', sa.Text(), nullable=True))
    for name in ('feature_version', 'features', 'fits_gzip'):
        op.alter_column('gong_snapshot', name, nullable=True)
    for name in ('cache_path', 'b0_deg', 'valid_fraction', 'carrington_lon'):
        op.alter_column('aia_snapshot', name, nullable=True)
    op.create_table('goes_snapshot',
        sa.Column('slot_at', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sha256', sa.String(64), nullable=False),
        sa.Column('source_product', sa.String(80), primary_key=True),
        sa.Column('raw_path', sa.Text(), nullable=False))
    op.create_index('ix_goes_snapshot_observed_at', 'goes_snapshot', ['observed_at'])


def downgrade():
    # New rows intentionally have no derived features; a downgrade cannot
    # reconstruct these without running model code or discarding observations.
    raise RuntimeError('Raw observation storage cannot be downgraded without a data migration')
