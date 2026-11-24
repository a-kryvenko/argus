"""Store the rolling verification responses prepared by the worker."""
from alembic import op

revision = '20261124_verification_summary'
down_revision = '20261027_drop_current_forecast'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE prophet.forecast_verification_summary (
        product text PRIMARY KEY,
        as_of timestamptz NOT NULL,
        report jsonb NOT NULL
    )""")


def downgrade():
    op.drop_table('forecast_verification_summary', schema='prophet')
