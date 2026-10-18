"""Requeue legacy partial aggregates for closed-window processing."""
from alembic import op

revision = '20260908_0008'
down_revision = '20260908_0007'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        INSERT INTO solar_wind_aggregate_pending (kind, hour)
        SELECT DISTINCT kind, date_trunc('hour', bucket_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
        FROM solar_wind_aggregate WHERE statistics->>'window_complete' = 'false'
        ON CONFLICT (kind, hour) DO UPDATE SET hour = EXCLUDED.hour
    """)


def downgrade():
    # Queued work is valid under both worker versions; no data/schema to undo.
    pass
