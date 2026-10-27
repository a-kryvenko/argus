"""Remove unused measurement metadata and derive fixed interval boundaries."""
from alembic import op
import sqlalchemy as sa

revision = '20261027_measurement_metadata'
down_revision = '20261027_drop_legacy_files'
branch_labels = None
depends_on = None
online_compatible = False


def upgrade():
    for name in ('interval_end', 'spacecraft', 'provider_quality', 'station_count'):
        op.drop_column('measurement', name, schema='clio')


def downgrade():
    op.add_column('measurement', sa.Column('interval_end', sa.DateTime(timezone=True)), schema='clio')
    op.add_column('measurement', sa.Column('spacecraft', sa.String(32)), schema='clio')
    op.add_column('measurement', sa.Column('provider_quality', sa.Integer()), schema='clio')
    op.add_column('measurement', sa.Column('station_count', sa.Integer()), schema='clio')
    op.execute("""
        UPDATE clio.measurement SET interval_end = observed_at +
            CASE WHEN metric IN ('kp', 'ap') THEN interval '3 hours' ELSE interval '1 hour' END
        WHERE metric IN ('kp', 'ap', 'dst')
    """)
