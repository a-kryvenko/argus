"""Clear rejected values while retaining the column for older HTTP processes."""
from alembic import op

revision = '20261102_provider_contract'
down_revision = '20261101_drop_source_status'
branch_labels = None
depends_on = None
online_compatible = True


def upgrade():
    op.execute("""UPDATE clio.measurement SET value = NULL, quality = NULL
                  WHERE quality IN ('flagged', 'missing')""")


def downgrade():
    pass
