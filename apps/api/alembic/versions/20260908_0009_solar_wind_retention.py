"""Protect aggregates after verified raw-hour retirement."""
from alembic import op
import sqlalchemy as sa

revision = '20260908_0009'
down_revision = '20260908_0008'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('solar_wind_retired_hour',
        sa.Column('kind', sa.String(16), primary_key=True),
        sa.Column('hour', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('raw_rows', sa.Integer(), nullable=False))
    op.execute("""
        CREATE FUNCTION protect_retired_solar_wind() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE target_hour timestamptz;
        BEGIN
          IF TG_TABLE_NAME = 'solar_wind_aggregate_pending' THEN
            target_hour := NEW.hour;
          ELSE
            target_hour := date_trunc('hour', NEW.observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
            IF TG_OP = 'UPDATE' AND EXISTS (
              SELECT 1 FROM solar_wind_retired_hour WHERE kind=OLD.kind
              AND hour=date_trunc('hour', OLD.observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
            ) THEN
              RAISE EXCEPTION 'Raw solar wind hour has been retired';
            END IF;
          END IF;
          IF EXISTS (SELECT 1 FROM solar_wind_retired_hour WHERE kind=NEW.kind AND hour=target_hour) THEN
            RAISE EXCEPTION 'Raw solar wind hour has been retired';
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER protect_retired_solar_wind_raw
          BEFORE INSERT OR UPDATE ON solar_wind_observation
          FOR EACH ROW EXECUTE FUNCTION protect_retired_solar_wind();
        CREATE TRIGGER protect_retired_solar_wind_queue
          BEFORE INSERT OR UPDATE ON solar_wind_aggregate_pending
          FOR EACH ROW EXECUTE FUNCTION protect_retired_solar_wind();
        CREATE OR REPLACE FUNCTION queue_solar_wind_aggregate() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' AND EXISTS (
            SELECT 1 FROM solar_wind_retired_hour WHERE kind=OLD.kind
            AND hour=date_trunc('hour', OLD.observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
          ) THEN
            RETURN NULL;
          END IF;
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
    """)


def downgrade():
    # Removing protection after actual cleanup would allow destructive rebuilds.
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM solar_wind_retired_hour) THEN
          RAISE EXCEPTION 'Cannot remove retention protection after raw hours were retired';
        END IF;
    END $$;""")
    op.execute('DROP TRIGGER protect_retired_solar_wind_raw ON solar_wind_observation')
    op.execute('DROP TRIGGER protect_retired_solar_wind_queue ON solar_wind_aggregate_pending')
    op.execute('DROP FUNCTION protect_retired_solar_wind()')
    op.execute("""
        CREATE OR REPLACE FUNCTION queue_solar_wind_aggregate() RETURNS trigger LANGUAGE plpgsql AS $$
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
    """)
    op.drop_table('solar_wind_retired_hour')
