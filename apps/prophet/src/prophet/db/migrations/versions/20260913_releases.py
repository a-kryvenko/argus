"""Published product releases and retryable CSV exports."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = '20260913_prophet_releases'
down_revision = '20260913_prophet_runs'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('forecast_release',
        sa.Column('id', pg.UUID(as_uuid=True), primary_key=True),
        sa.Column('product', sa.String(64), nullable=False),
        sa.Column('run_id', pg.UUID(as_uuid=True), sa.ForeignKey('forecast_run.id'), nullable=False),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('issue_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('artifact_names', pg.JSONB(), nullable=False),
        sa.UniqueConstraint('product', 'run_id'),
        sa.UniqueConstraint('product', 'id'))
    op.create_table('current_forecast',
        sa.Column('product', sa.String(64), primary_key=True),
        sa.Column('release_id', pg.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(['product', 'release_id'], ['forecast_release.product', 'forecast_release.id']))
    op.create_table('forecast_export',
        sa.Column('release_id', pg.UUID(as_uuid=True), sa.ForeignKey('forecast_release.id'), primary_key=True),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('exported_at', sa.DateTime(timezone=True)),
        sa.Column('error', sa.Text()))


def downgrade():
    op.drop_table('forecast_export')
    op.drop_table('current_forecast')
    op.drop_table('forecast_release')
