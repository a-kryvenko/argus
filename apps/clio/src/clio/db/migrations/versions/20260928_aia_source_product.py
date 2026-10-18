"""Record the product supplying newly collected AIA files."""
from alembic import op
import sqlalchemy as sa

revision = '20260928_aia_source_product'
down_revision = '20260928_measurement_provenance'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('aia_snapshot', sa.Column('source_product', sa.String(80), nullable=True))


def downgrade():
    op.drop_column('aia_snapshot', 'source_product')
