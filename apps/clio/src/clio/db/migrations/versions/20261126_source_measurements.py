"""Identify numeric readings by source and mark model input time basis."""
from alembic import op
import sqlalchemy as sa

revision = '20261126_source_measurements'
down_revision = '20261107_event_identities'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('measurement', sa.Column('spacecraft', sa.String(32)), schema='clio')
    op.drop_constraint('uq_measurement_metric', 'measurement', type_='unique', schema='clio')
    op.create_unique_constraint('uq_measurement_metric', 'measurement',
        ['metric', 'observed_at', 'source_product'], postgresql_nulls_not_distinct=True, schema='clio')
    op.create_index('ix_measurement_metric_source_time', 'measurement',
        ['metric', 'source_product', 'observed_at'], schema='clio')
    op.add_column('normalized_observation', sa.Column('wind_time_basis', sa.String(16)), schema='clio')


def downgrade():
    op.drop_index('ix_measurement_metric_source_time', table_name='measurement', schema='clio')
    op.drop_constraint('uq_measurement_metric', 'measurement', type_='unique', schema='clio')
    # PostgreSQL refuses this downgrade if observations from multiple products now overlap.
    op.create_unique_constraint('uq_measurement_metric', 'measurement',
        ['metric', 'observed_at'], schema='clio')
    op.drop_column('measurement', 'spacecraft', schema='clio')
    op.drop_column('normalized_observation', 'wind_time_basis', schema='clio')
