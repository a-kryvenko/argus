"""Remove collection attempt diagnostics; observation freshness uses stored data."""
from alembic import op
import sqlalchemy as sa

revision = '20261101_drop_source_status'
down_revision = '20261027_measurement_metadata'
branch_labels = None
depends_on = None
# Older HTTP processes still read this table.
online_compatible = False


def upgrade():
    op.drop_table('observation_source_status', schema='clio')


def downgrade():
    op.create_table('observation_source_status',
        sa.Column('source_id', sa.String(32), primary_key=True),
        sa.Column('last_attempt_at', sa.DateTime(timezone=True), nullable=False),
        *[sa.Column(name, sa.DateTime(timezone=True)) for name in (
            'last_completed_at', 'last_response_at', 'last_success_at', 'last_error_at',
            'latest_observation_at', 'latest_interval_end')],
        sa.Column('last_error_code', sa.String(32)),
        sa.Column('last_error_message', sa.String(160)),
        sa.Column('consecutive_failures', sa.Integer(), nullable=False),
        sa.Column('data_quality', sa.String(16)),
        schema='clio',
    )
