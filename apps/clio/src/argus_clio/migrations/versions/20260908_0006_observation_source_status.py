"""Persist collection attempts, outcomes and source freshness."""
from alembic import op
import sqlalchemy as sa

revision = '20260908_0006'
down_revision = '20260907_0005'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('observation_source_status',
        sa.Column('source_id', sa.String(32), primary_key=True),
        sa.Column('last_attempt_at', sa.DateTime(timezone=True), nullable=False),
        *[sa.Column(name, sa.DateTime(timezone=True), nullable=True) for name in (
            'last_completed_at', 'last_response_at', 'last_success_at', 'last_error_at',
            'latest_observation_at', 'latest_interval_end')],
        sa.Column('last_error_code', sa.String(32), nullable=True),
        sa.Column('last_error_message', sa.String(160), nullable=True),
        sa.Column('consecutive_failures', sa.Integer(), nullable=False),
        sa.Column('data_quality', sa.String(16), nullable=True),
    )


def downgrade():
    op.drop_table('observation_source_status')
