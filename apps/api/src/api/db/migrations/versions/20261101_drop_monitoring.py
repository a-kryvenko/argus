"""Remove persisted monitoring and request statistics."""
from alembic import op
import sqlalchemy as sa

revision = '20261101_drop_monitoring'
down_revision = '20261001_0012'
branch_labels = None
depends_on = None
# Older API processes still read these tables.
online_compatible = True


def upgrade():
    op.drop_table('traffic_metric', schema='api')
    op.drop_table('monitor_state', schema='api')
    op.drop_table('api_metric', schema='api')
    op.execute("""UPDATE api.dashboard_group
        SET permissions = (permissions::jsonb - 'api_stats.read')::json
        WHERE permissions::jsonb ? 'api_stats.read'""")


def downgrade():
    op.create_table('api_metric',
        sa.Column('hour', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('route', sa.String(256), primary_key=True),
        sa.Column('method', sa.String(16), primary_key=True),
        sa.Column('status', sa.Integer(), primary_key=True),
        sa.Column('bucket_ms', sa.Integer(), primary_key=True),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.Column('duration_ms', sa.Float(), nullable=False),
        schema='api',
    )
    op.execute("""UPDATE api.dashboard_group
        SET permissions = (permissions::jsonb || '["api_stats.read"]'::jsonb)::json
        WHERE name = 'admins' AND NOT permissions::jsonb ? 'api_stats.read'""")
    op.create_table('monitor_state',
        sa.Column('name', sa.String(32), primary_key=True),
        sa.Column('checked_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        schema='api',
    )
    op.create_table('traffic_metric',
        sa.Column('time', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('resolution', sa.String(8), primary_key=True),
        sa.Column('channel', sa.String(8), primary_key=True),
        sa.Column('status', sa.Integer(), primary_key=True),
        sa.Column('bucket_ms', sa.Integer(), primary_key=True),
        sa.Column('count', sa.BigInteger(), nullable=False),
        sa.Column('duration_ms', sa.Float(), nullable=False),
        schema='api',
    )
