"""Store native minute solar wind observations separately from model inputs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260907_0004"
down_revision = "20260905_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "solar_wind_observation",
        sa.Column("kind", sa.String(16), primary_key=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("spacecraft", sa.String(32), primary_key=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("values", postgresql.JSONB(), nullable=False),
        sa.Column("raw", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_solar_wind_active_time", "solar_wind_observation", ["kind", "observed_at"], postgresql_where=sa.text("active"))


def downgrade():
    op.drop_table("solar_wind_observation")
