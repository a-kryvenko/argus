"""Remove the retired AIA collector table; originals belong to the SDO archive."""
from alembic import op

revision = '20261004_drop_aia_snapshot'
down_revision = '20261004_raw_observation_files'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table('aia_snapshot')


def downgrade():
    raise RuntimeError('Deleted AIA snapshot history can only be restored from a backup')
