"""Remove the retired dashboard observation-browser permission."""
from alembic import op

revision = '20261001_0012'
down_revision = '20260917_0011'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""UPDATE dashboard_group
        SET permissions = (permissions::jsonb - 'observations.read')::json
        WHERE permissions::jsonb ? 'observations.read'""")


def downgrade():
    # Restore the original seeded admin permission; custom grants stay revoked.
    op.execute("""UPDATE dashboard_group
        SET permissions = (permissions::jsonb || '["observations.read"]'::jsonb)::json
        WHERE name = 'admins' AND NOT permissions::jsonb ? 'observations.read'""")
