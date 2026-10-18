"""Remove obsolete filesystem export tracking; stored forecast bytes are retained."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = '20260918_prophet_no_exports'
down_revision = '20260913_prophet_scheduler'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table('forecast_export')
    op.drop_column('forecast_artifact', 'csv_written_at')


def downgrade():
    op.add_column('forecast_artifact', sa.Column('csv_written_at', sa.DateTime(timezone=True)))
    op.create_table('forecast_export',
        sa.Column('release_id', pg.UUID(as_uuid=True), sa.ForeignKey('forecast_release.id'), primary_key=True),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('exported_at', sa.DateTime(timezone=True)),
        sa.Column('error', sa.Text()))
    # A reverted exporter can reconstruct files from retained releases.
    op.execute('INSERT INTO forecast_export (release_id) SELECT id FROM forecast_release')
