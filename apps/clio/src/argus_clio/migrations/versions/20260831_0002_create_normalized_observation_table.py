"""Create normalized observation table.

Revision ID: 20260831_0002
Revises: 20260830_0001
Create Date: 2026-08-31
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260831_0002"
down_revision: str | None = "20260830_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "normalized_observation",
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("bx", sa.Double(), nullable=False),
        sa.Column("by", sa.Double(), nullable=False),
        sa.Column("bz", sa.Double(), nullable=False),
        sa.Column("v", sa.Double(), nullable=False),
        sa.Column("n", sa.Double(), nullable=False),
        sa.Column("t", sa.Double(), nullable=False),
        sa.Column("kp", sa.Double(), nullable=False),
        sa.Column("dst", sa.Double(), nullable=False),
        sa.Column("ap", sa.Double(), nullable=False),
        sa.Column("f10_7", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint(
            "observed_at",
            name=op.f("pk_normalized_observation"),
        ),
    )


def downgrade() -> None:
    op.drop_table("normalized_observation")
