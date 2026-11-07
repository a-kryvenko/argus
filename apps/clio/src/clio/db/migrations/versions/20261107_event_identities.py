"""Add stable source identities to normalized file observations."""
from alembic import op
import sqlalchemy as sa

revision = '20261107_event_identities'
down_revision = '20261107_file_observation'
branch_labels = None
depends_on = None
online_compatible = True


def upgrade():
    op.add_column('file_observation', sa.Column('source_id', sa.Text()), schema='clio')
    op.drop_constraint('uq_file_observation_original', 'file_observation', type_='unique', schema='clio')
    op.create_index('uq_file_observation_original', 'file_observation', ['observation', 'file_id'],
        unique=True, postgresql_where=sa.text('source_id IS NULL'), schema='clio')
    op.create_unique_constraint('uq_file_observation_source_id', 'file_observation',
        ['observation', 'source_id'], schema='clio')


def downgrade():
    raise RuntimeError('Keep stable event identities while a deployed release may use them')
