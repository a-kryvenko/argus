"""Persist the latest DONKI attempt independently of a successful receipt."""
from alembic import op
import sqlalchemy as sa

revision = '20261126_donki_poll_status'
down_revision = '20261126_source_measurements'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'donki_poll_status',
        sa.Column('endpoint', sa.String(32), primary_key=True),
        sa.Column('attempted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('error', sa.String(128)),
        schema='clio',
    )


def downgrade():
    op.drop_table('donki_poll_status', schema='clio')
