"""Latest reproducible verification evidence for each published artifact."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = '20260926_prophet_verification'
down_revision = '20260918_prophet_no_exports'
branch_labels = depends_on = None


def upgrade():
    op.create_table('forecast_verification',
        sa.Column('release_id', pg.UUID(as_uuid=True), sa.ForeignKey('forecast_release.id'), primary_key=True),
        sa.Column('artifact', sa.Text(), primary_key=True),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('report', pg.JSONB(), nullable=False))


def downgrade():
    op.drop_table('forecast_verification')
