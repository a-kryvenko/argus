"""Nullable provenance for newly adapted observations; old provenance is unknown."""
from alembic import op
import sqlalchemy as sa

revision = '20260928_measurement_provenance'
down_revision = '20260923_aia_snapshot'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('measurement', sa.Column('source_product', sa.String(80), nullable=True))
    op.add_column('measurement', sa.Column('received_at', sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column('measurement', 'received_at')
    op.drop_column('measurement', 'source_product')
