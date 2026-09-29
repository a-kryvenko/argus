"""Immutable GONG originals and features for saved forecast inputs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20260929_gong_snapshot'
down_revision = '20260928_aia_source_product'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('gong_snapshot',
        sa.Column('slot_at', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sha256', sa.String(64), nullable=False),
        sa.Column('source_product', sa.String(80), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=False),
        sa.Column('feature_version', sa.String(40), nullable=False),
        sa.Column('features', postgresql.JSONB(), nullable=False),
        sa.Column('fits_gzip', sa.LargeBinary(), nullable=False))
    op.create_index('ix_gong_snapshot_observed_at', 'gong_snapshot', ['observed_at'])


def downgrade():
    op.drop_table('gong_snapshot')
