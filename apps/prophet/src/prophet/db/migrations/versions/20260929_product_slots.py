"""Product-owned schedule slots, retaining historical aggregate slots as all."""
from alembic import op
import sqlalchemy as sa

revision = '20260929_prophet_product_slots'
down_revision = '20260929_prophet_jobs'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('forecast_slot', sa.Column('product', sa.String(32), nullable=False, server_default='all'))
    op.add_column('forecast_run', sa.Column('scheduled_product', sa.String(32)))
    op.execute("UPDATE forecast_run SET scheduled_product='all' WHERE scheduled_slot IS NOT NULL")
    op.drop_constraint('fk_forecast_run_scheduled_slot', 'forecast_run', type_='foreignkey')
    op.drop_constraint('forecast_slot_pkey', 'forecast_slot', type_='primary')
    op.drop_constraint('slot_hour', 'forecast_slot', type_='check')
    op.create_primary_key('forecast_slot_pkey', 'forecast_slot', ['product', 'slot'])
    op.create_foreign_key('fk_forecast_run_scheduled_slot', 'forecast_run', 'forecast_slot',
                         ['scheduled_product', 'scheduled_slot'], ['product', 'slot'])
    op.create_check_constraint('run_schedule_pair', 'forecast_run',
                               '(scheduled_slot IS NULL) = (scheduled_product IS NULL)')


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM forecast_slot WHERE product<>'all')")):
        raise RuntimeError('Cannot collapse product schedules after product slots have been recorded')
    op.drop_constraint('run_schedule_pair', 'forecast_run', type_='check')
    op.drop_constraint('fk_forecast_run_scheduled_slot', 'forecast_run', type_='foreignkey')
    op.drop_column('forecast_run', 'scheduled_product')
    op.drop_constraint('forecast_slot_pkey', 'forecast_slot', type_='primary')
    op.drop_column('forecast_slot', 'product')
    op.create_primary_key('forecast_slot_pkey', 'forecast_slot', ['slot'])
    op.create_check_constraint('slot_hour', 'forecast_slot',
                               "slot AT TIME ZONE 'UTC' = date_trunc('hour', slot AT TIME ZONE 'UTC')")
    op.create_foreign_key('fk_forecast_run_scheduled_slot', 'forecast_run', 'forecast_slot', ['scheduled_slot'], ['slot'])
