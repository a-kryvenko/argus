"""Remove completion markers unused by cron commands; preserve forecast history."""
from alembic import op
from alembic.util import CommandError

revision = '20261027_drop_scheduler_jobs'
down_revision = '20260929_prophet_product_slots'
branch_labels = None
depends_on = None
online_compatible = True


def upgrade():
    op.drop_table('scheduled_job', schema='prophet')


def downgrade():
    raise CommandError('Retired scheduler markers require restoring a backup')
