"""Versioned solar wind aggregates and durable recalculation queue."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20260908_0007'
down_revision = '20260908_0006'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('solar_wind_aggregate',
        sa.Column('kind', sa.String(16), primary_key=True),
        sa.Column('resolution_seconds', sa.Integer(), primary_key=True),
        sa.Column('bucket_start', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('calculated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('statistics', postgresql.JSONB(), nullable=False))
    op.create_table('solar_wind_aggregate_pending',
        sa.Column('kind', sa.String(16), primary_key=True),
        sa.Column('hour', sa.DateTime(timezone=True), primary_key=True))
    # Enqueue in the same transaction as raw changes, including updates by older
    # collector processes. UTC alignment is independent of connection timezone.
    op.execute("""
        CREATE FUNCTION queue_solar_wind_aggregate() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP <> 'INSERT' THEN
            INSERT INTO solar_wind_aggregate_pending VALUES
              (OLD.kind, date_trunc('hour', OLD.observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
              ON CONFLICT (kind, hour) DO UPDATE SET hour = EXCLUDED.hour;
          END IF;
          IF TG_OP <> 'DELETE' THEN
            INSERT INTO solar_wind_aggregate_pending VALUES
              (NEW.kind, date_trunc('hour', NEW.observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
              ON CONFLICT (kind, hour) DO UPDATE SET hour = EXCLUDED.hour;
          END IF;
          RETURN NULL;
        END $$;
        CREATE TRIGGER solar_wind_aggregate_changed
        AFTER INSERT OR UPDATE OR DELETE ON solar_wind_observation
        FOR EACH ROW EXECUTE FUNCTION queue_solar_wind_aggregate();
        INSERT INTO solar_wind_aggregate_pending
        SELECT DISTINCT kind, date_trunc('hour', observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
        FROM solar_wind_observation ON CONFLICT (kind, hour) DO UPDATE SET hour = EXCLUDED.hour;
    """)


def downgrade():
    op.execute('DROP TRIGGER solar_wind_aggregate_changed ON solar_wind_observation')
    op.execute('DROP FUNCTION queue_solar_wind_aggregate()')
    op.drop_table('solar_wind_aggregate_pending')
    op.drop_table('solar_wind_aggregate')
