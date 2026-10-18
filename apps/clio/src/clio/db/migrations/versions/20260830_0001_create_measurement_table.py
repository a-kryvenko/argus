"""Create measurement table.

Revision ID: 20260830_0001
Revises:
Create Date: 2026-08-30
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260830_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "measurement",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("metric", sa.String(length=16), nullable=False),
        sa.Column("value", sa.Double(), nullable=False),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_measurement")),
        sa.UniqueConstraint(
            "metric",
            "observed_at",
            name=op.f("uq_measurement_metric"),
        ),
    )
    op.create_index(
        op.f("ix_measurement_observed_at"),
        "measurement",
        ["observed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_measurement_observed_at"),
        table_name="measurement",
    )
    op.drop_table("measurement")
