"""Store native estimated Kp and real-time Dst intervals."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20260907_0005'
down_revision = '20260907_0004'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('geomagnetic_observation',
        sa.Column('metric', sa.String(16), primary_key=True),
        sa.Column('interval_start', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('interval_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('value', sa.Double(), nullable=True),
        sa.Column('quality', sa.String(16), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('raw', postgresql.JSONB(), nullable=False),
    )


def downgrade():
    op.drop_table('geomagnetic_observation')
