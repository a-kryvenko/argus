"""Receipt timestamps per model-input metric; historical receipt times are unknown."""
from alembic import op
import sqlalchemy as sa
revision = '20260917_measurement_receipt'
down_revision = '20260912_clio_jobs'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('measurement_receipt', sa.Column('metric', sa.String(16), primary_key=True),
        sa.Column('latest_observation_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False))

def downgrade():
    op.drop_table('measurement_receipt')
