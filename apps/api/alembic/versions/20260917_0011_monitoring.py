"""Project monitoring permissions, snapshots and traffic."""
from alembic import op
import sqlalchemy as sa
revision = '20260917_0011'
down_revision = '20260911_0010'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('monitor_state', sa.Column('name', sa.String(32), primary_key=True),
        sa.Column('checked_at', sa.DateTime(timezone=True), nullable=False), sa.Column('payload', sa.JSON, nullable=False))
    op.create_table('traffic_metric',
        sa.Column('time', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('resolution', sa.String(8), primary_key=True),
        sa.Column('channel', sa.String(8), primary_key=True),
        sa.Column('status', sa.Integer, primary_key=True),
        sa.Column('bucket_ms', sa.Integer, primary_key=True),
        sa.Column('count', sa.BigInteger, nullable=False), sa.Column('duration_ms', sa.Float, nullable=False))
    op.execute("""UPDATE dashboard_group SET permissions =
        (permissions::jsonb || '["project_monitoring.read"]'::jsonb)::json
        WHERE name = 'admins' AND NOT permissions::jsonb ? 'project_monitoring.read'""")
    op.execute("INSERT INTO dashboard_group(name, permissions) VALUES ('clients', '[]'::json) ON CONFLICT DO NOTHING")

def downgrade():
    op.execute("UPDATE dashboard_group SET permissions = (permissions::jsonb - 'project_monitoring.read')::json")
    op.drop_table('traffic_metric')
    op.drop_table('monitor_state')
