"""Owned scheduler progress and schema-bound retention/aggregation triggers."""
from alembic import op
import sqlalchemy as sa

revision = '20260912_clio_jobs'
down_revision = '20260908_0009'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('scheduled_job',
        sa.Column('name', sa.String(32), primary_key=True),
        sa.Column('completed_slot', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=False))
    for function in ('queue_solar_wind_aggregate', 'protect_retired_solar_wind'):
        op.execute(f'ALTER FUNCTION clio.{function}() SET search_path TO clio, pg_catalog, pg_temp')


def downgrade():
    op.drop_table('scheduled_job')
