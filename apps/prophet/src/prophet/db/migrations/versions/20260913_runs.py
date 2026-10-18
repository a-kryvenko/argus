"""Forecast execution records, immutable input snapshots and captured CSV results."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = '20260913_prophet_runs'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('forecast_run',
        sa.Column('id', pg.UUID(as_uuid=True), primary_key=True),
        sa.Column('scope', sa.Text(), nullable=False),
        sa.Column('product', sa.String(32), nullable=False),
        sa.Column('trigger', sa.String(16), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True)),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('input_snapshot', pg.JSONB()),
        sa.Column('input_sha256', sa.String(64)),
        sa.Column('provenance', pg.JSONB(), nullable=False),
        sa.Column('error', sa.Text()),
        sa.CheckConstraint("status IN ('running','succeeded','partial','failed','interrupted')", name='run_status'),
        sa.CheckConstraint("trigger IN ('manual','scheduled')", name='run_trigger'))
    op.create_index('ix_forecast_run_started_at', 'forecast_run', ['started_at'])
    op.create_index('ix_forecast_run_scope_status', 'forecast_run', ['scope', 'status'])
    op.create_table('forecast_artifact',
        sa.Column('run_id', pg.UUID(as_uuid=True), sa.ForeignKey('forecast_run.id'), primary_key=True),
        sa.Column('name', sa.String(100), primary_key=True),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('csv_written_at', sa.DateTime(timezone=True)),
        sa.Column('csv_gzip', sa.LargeBinary()),
        sa.Column('sha256', sa.String(64)),
        sa.Column('row_count', sa.BigInteger()),
        sa.Column('columns', pg.JSONB()),
        sa.Column('model_info', pg.JSONB(), nullable=False),
        sa.Column('error', sa.Text()),
        sa.CheckConstraint("status IN ('stored','skipped')", name='artifact_status'),
        sa.CheckConstraint("(status='stored' AND csv_gzip IS NOT NULL AND sha256 IS NOT NULL AND row_count > 0) OR (status='skipped' AND csv_gzip IS NULL)", name='artifact_payload'))


def downgrade():
    op.drop_table('forecast_artifact')
    op.drop_table('forecast_run')
