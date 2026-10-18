"""Consolidate native observations and keep measurement as permanent history."""
from alembic import op
import sqlalchemy as sa

revision = '20261003_unified_measurement'
down_revision = '20260929_gong_snapshot'
branch_labels = None
depends_on = None


def upgrade():
    # Readers and collectors must be stopped for this schema transition.
    op.add_column('measurement', sa.Column('interval_end', sa.DateTime(timezone=True)))
    op.add_column('measurement', sa.Column('quality', sa.String(16)))
    op.add_column('measurement', sa.Column('spacecraft', sa.String(32)))
    op.add_column('measurement', sa.Column('provider_quality', sa.Integer()))
    op.add_column('measurement', sa.Column('station_count', sa.Integer()))
    op.alter_column('measurement', 'value', existing_type=sa.Double(), nullable=True)
    op.execute("""
        UPDATE measurement m SET received_at = r.received_at
        FROM measurement_receipt r
        WHERE m.metric = r.metric AND m.observed_at = r.latest_observation_at AND m.received_at IS NULL
    """)
    # One NOAA active spacecraft per timestamp, matching the previous read selection.
    op.execute("""
        INSERT INTO measurement(metric, observed_at, value, source_product, received_at,
                                spacecraft, provider_quality, quality)
        SELECT v.key, s.observed_at, (v.value #>> '{}')::double precision,
               'swpc.rtsw_' || s.kind, s.received_at, s.spacecraft,
               (s.raw->>'overall_quality')::integer,
               CASE WHEN v.value = 'null'::jsonb THEN 'missing'
                    WHEN coalesce((s.raw->>'overall_quality')::integer, 0) <> 0 THEN 'flagged'
                    ELSE 'unverified' END
        FROM (SELECT DISTINCT ON (kind, observed_at) * FROM solar_wind_observation
              WHERE active ORDER BY kind, observed_at, received_at DESC, spacecraft) s
        CROSS JOIN LATERAL jsonb_each(s.values) v
        WHERE v.key IN ('bx','by','bz','bt','v','n','t')
        ON CONFLICT (metric, observed_at) DO UPDATE SET
            value=EXCLUDED.value, source_product=EXCLUDED.source_product, received_at=EXCLUDED.received_at,
            spacecraft=EXCLUDED.spacecraft, provider_quality=EXCLUDED.provider_quality, quality=EXCLUDED.quality
    """)
    op.execute("""
        INSERT INTO measurement(metric, observed_at, value, source_product, received_at, interval_end, quality, station_count)
        SELECT metric, interval_start, value, 'swpc.' || metric, received_at, interval_end, quality,
               (raw->>'station_count')::integer FROM geomagnetic_observation
        ON CONFLICT (metric, observed_at) DO UPDATE SET
            value=EXCLUDED.value, source_product=EXCLUDED.source_product, received_at=EXCLUDED.received_at,
            interval_end=EXCLUDED.interval_end, quality=EXCLUDED.quality, station_count=EXCLUDED.station_count
    """)
    op.execute("""
        INSERT INTO measurement(metric, observed_at, value, source_product, received_at, interval_end, quality, station_count)
        SELECT 'ap', interval_start, ap, 'swpc.kp', received_at, interval_end,
               CASE WHEN ap IS NULL THEN 'missing' ELSE quality END, (raw->>'station_count')::integer
        FROM (SELECT *, CASE WHEN raw->>'a_running' ~ '^[+]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]{1,2})?$'
                             THEN (raw->>'a_running')::double precision ELSE NULL END AS ap
              FROM geomagnetic_observation WHERE metric='kp') g
        ON CONFLICT (metric, observed_at) DO UPDATE SET
            value=EXCLUDED.value, source_product=EXCLUDED.source_product, received_at=EXCLUDED.received_at,
            interval_end=EXCLUDED.interval_end, quality=EXCLUDED.quality, station_count=EXCLUDED.station_count
    """)
    op.drop_table('solar_wind_observation')  # Also removes its triggers.
    op.drop_table('solar_wind_aggregate_pending')
    op.execute('DROP FUNCTION queue_solar_wind_aggregate()')
    op.execute('DROP FUNCTION protect_retired_solar_wind()')
    op.drop_table('solar_wind_retired_hour')
    op.drop_table('solar_wind_aggregate')
    op.drop_table('geomagnetic_observation')
    op.drop_table('measurement_receipt')
    op.execute("DELETE FROM scheduled_job WHERE name = 'aggregate'")


def downgrade():
    raise RuntimeError('Restore a pre-migration backup to recover discarded source payloads and aggregate-only history')
