"""Add optional calibrated solar indices to observations.

Revision ID: 20260905_0003
Revises: 20260831_0002
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260905_0003"
down_revision: str | None = "20260831_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for name in ("s10", "m10", "y10"):
        op.add_column("normalized_observation", sa.Column(name, sa.Double(), nullable=True))


def downgrade() -> None:
    for name in ("y10", "m10", "s10"):
        op.drop_column("normalized_observation", name)
