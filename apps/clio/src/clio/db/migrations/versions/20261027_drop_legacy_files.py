"""Remove retired file storage and scheduler markers after the file-only release."""
from alembic import op
from alembic.util import CommandError

revision = '20261027_drop_legacy_files'
down_revision = '20261027_file_cutover'
branch_labels = None
depends_on = None
online_compatible = True


def upgrade():
    for table in ('gong_snapshot', 'goes_snapshot', 'donki_poll', 'enlil_profile_receipt', 'scheduled_job'):
        op.drop_table(table, schema='clio')


def downgrade():
    raise CommandError('Legacy payloads cannot be reconstructed by a schema downgrade; restore a backup')
