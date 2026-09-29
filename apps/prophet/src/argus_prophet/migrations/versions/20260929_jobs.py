"""Durable completion markers for scheduled verification."""
from alembic import op
import sqlalchemy as sa

revision = '20260929_prophet_jobs'
down_revision = '20260926_prophet_verification'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('scheduled_job',
        sa.Column('name', sa.Text(), primary_key=True),
        sa.Column('completed_slot', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=False))


def downgrade():
    op.drop_table('scheduled_job')
