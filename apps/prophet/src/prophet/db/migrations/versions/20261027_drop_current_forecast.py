"""Remove the legacy pointer after readers switched to forecast_release."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = '20261027_drop_current_forecast'
down_revision = '20261027_drop_forecast_slots'
branch_labels = None
depends_on = None
# Deployment drains old workers before migration; released HTTP readers no longer use the pointer.
online_compatible = True


def upgrade():
    op.drop_table('current_forecast', schema='prophet')


def downgrade():
    op.create_table('current_forecast',
        sa.Column('product', sa.String(64), primary_key=True),
        sa.Column('release_id', pg.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(['product', 'release_id'],
                                ['prophet.forecast_release.product', 'prophet.forecast_release.id']),
        schema='prophet')
    op.execute('''INSERT INTO prophet.current_forecast(product,release_id)
        SELECT DISTINCT ON (r.product) r.product,r.id
        FROM prophet.forecast_release r JOIN prophet.forecast_run f ON f.id=r.run_id
        ORDER BY r.product,r.issue_time DESC,f.started_at DESC,r.published_at DESC,r.id DESC''')
