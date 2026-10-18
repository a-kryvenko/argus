"""Database-owned hourly slots linked to forecast attempts."""
from alembic import op
import sqlalchemy as sa

revision = '20260913_prophet_scheduler'
down_revision = '20260913_prophet_releases'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('forecast_slot',
        sa.Column('slot', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('finished_at', sa.DateTime(timezone=True)),
        sa.Column('error', sa.Text()),
        sa.CheckConstraint("status IN ('running','succeeded','partial','failed','interrupted','imported')", name='slot_status'),
        sa.CheckConstraint("(status='imported' AND attempts=0) OR (status<>'imported' AND attempts>0)", name='slot_attempts'),
        sa.CheckConstraint("slot AT TIME ZONE 'UTC' = date_trunc('hour', slot AT TIME ZONE 'UTC')", name='slot_hour'))
    op.create_index('ix_forecast_slot_status_slot', 'forecast_slot', ['status', 'slot'])
    op.add_column('forecast_run', sa.Column('scheduled_slot', sa.DateTime(timezone=True)))
    op.create_foreign_key('fk_forecast_run_scheduled_slot', 'forecast_run', 'forecast_slot', ['scheduled_slot'], ['slot'])
    op.create_index('ix_forecast_run_scheduled_slot', 'forecast_run', ['scheduled_slot'])
    op.create_check_constraint('run_scheduled_slot_trigger', 'forecast_run', "scheduled_slot IS NULL OR trigger='scheduled'")


def downgrade():
    op.drop_constraint('run_scheduled_slot_trigger', 'forecast_run', type_='check')
    op.drop_index('ix_forecast_run_scheduled_slot', table_name='forecast_run')
    op.drop_constraint('fk_forecast_run_scheduled_slot', 'forecast_run', type_='foreignkey')
    op.drop_column('forecast_run', 'scheduled_slot')
    op.drop_table('forecast_slot')
