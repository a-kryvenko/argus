"""Remove retired scheduler slots and their links from forecast history."""
from alembic import op
from alembic.util import CommandError

revision = '20261027_drop_forecast_slots'
down_revision = '20261027_drop_scheduler_jobs'
branch_labels = None
depends_on = None
online_compatible = True


def upgrade():
    op.drop_constraint('fk_forecast_run_scheduled_slot', 'forecast_run', schema='prophet', type_='foreignkey')
    op.drop_constraint('run_schedule_pair', 'forecast_run', schema='prophet', type_='check')
    op.drop_constraint('run_scheduled_slot_trigger', 'forecast_run', schema='prophet', type_='check')
    op.drop_index('ix_forecast_run_scheduled_slot', table_name='forecast_run', schema='prophet')
    op.drop_column('forecast_run', 'scheduled_slot', schema='prophet')
    op.drop_column('forecast_run', 'scheduled_product', schema='prophet')
    op.drop_table('forecast_slot', schema='prophet')


def downgrade():
    raise CommandError('Retired scheduler slots require restoring a backup')
