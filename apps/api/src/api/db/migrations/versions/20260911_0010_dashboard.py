"""Dashboard users, groups, sessions and API statistics."""
from alembic import op
import sqlalchemy as sa

revision = '20260911_0010'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('dashboard_user', sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('username', sa.String(80), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(256), nullable=False),
        sa.Column('active', sa.Boolean, nullable=False))
    op.create_table('dashboard_group', sa.Column('name', sa.String(80), primary_key=True),
        sa.Column('permissions', sa.JSON, nullable=False))
    op.create_table('dashboard_membership',
        sa.Column('user_id', sa.Integer, sa.ForeignKey('dashboard_user.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('group_name', sa.String(80), sa.ForeignKey('dashboard_group.name', ondelete='CASCADE'), primary_key=True))
    op.create_table('dashboard_session', sa.Column('token_hash', sa.String(64), primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('dashboard_user.id', ondelete='CASCADE'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_dashboard_session_user_id', 'dashboard_session', ['user_id'])
    op.create_index('ix_dashboard_session_expires_at', 'dashboard_session', ['expires_at'])
    op.create_table('dashboard_login_attempt', sa.Column('key', sa.String(64), primary_key=True),
        sa.Column('window', sa.DateTime(timezone=True), nullable=False), sa.Column('count', sa.Integer, nullable=False))
    op.create_index('ix_dashboard_login_attempt_window', 'dashboard_login_attempt', ['window'])
    op.create_table('api_metric', sa.Column('hour', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('route', sa.String(256), primary_key=True), sa.Column('method', sa.String(16), primary_key=True),
        sa.Column('status', sa.Integer, primary_key=True), sa.Column('bucket_ms', sa.Integer, primary_key=True),
        sa.Column('count', sa.Integer, nullable=False), sa.Column('duration_ms', sa.Float, nullable=False))
    op.execute("""INSERT INTO dashboard_group(name, permissions)
        VALUES ('admins', '["observations.read", "users.manage", "api_stats.read"]'::json)""")


def downgrade():
    for table in ['api_metric', 'dashboard_login_attempt', 'dashboard_session', 'dashboard_membership', 'dashboard_group', 'dashboard_user']:
        op.drop_table(table)
