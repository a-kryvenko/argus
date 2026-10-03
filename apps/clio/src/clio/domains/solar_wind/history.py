"""Compute closed-window statistics from measurement, without materialized copies."""
from datetime import UTC, datetime, timedelta
import math

from sqlalchemy import text
from clio.domains.solar_wind.observations import metadata


async def history(session, metrics, start, end, seconds, now=None):
    if seconds not in (300, 3600):
        raise ValueError('Supported windows: 300 or 3600 seconds')
    now = now or datetime.now(UTC)
    left = datetime.fromtimestamp(math.floor(start.timestamp()/seconds)*seconds, UTC)
    right = datetime.fromtimestamp(math.floor(min(end, now).timestamp()/seconds)*seconds, UTC)
    # Deduplicate each minute before counting coverage. Only selected native
    # RTSW samples participate; historical hourly products have a different cadence.
    rows = (await session.execute(text("""
        WITH minutes AS (
            SELECT DISTINCT ON (metric, date_trunc('minute', observed_at))
                metric, observed_at, received_at, value, quality, spacecraft,
                to_timestamp(floor(extract(epoch FROM observed_at) / :seconds) * :seconds) AS bucket
            FROM clio.measurement
            WHERE metric = ANY(:metrics)
              AND source_product IN ('swpc.rtsw_mag', 'swpc.rtsw_plasma')
              AND observed_at >= :start AND observed_at < :end
            ORDER BY metric, date_trunc('minute', observed_at), observed_at DESC
        ), samples AS (
            SELECT *, value IS NOT NULL AND quality IS DISTINCT FROM 'flagged'
                      AND quality IS DISTINCT FROM 'missing' AS usable,
                   lag(spacecraft) OVER (PARTITION BY metric, bucket ORDER BY observed_at) AS previous_spacecraft
            FROM minutes
        )
        SELECT metric, bucket, max(received_at) AS received_at,
               count(*) AS samples, count(*) FILTER (WHERE usable) AS count,
               avg(value) FILTER (WHERE usable) AS mean,
               min(value) FILTER (WHERE usable) AS min,
               max(value) FILTER (WHERE usable) AS max,
               count(*) FILTER (WHERE usable AND value < 0) AS negative_count,
               (array_agg(spacecraft ORDER BY observed_at))[1] AS spacecraft,
               (array_agg(spacecraft ORDER BY observed_at DESC))[1] AS last_spacecraft,
               count(*) FILTER (WHERE previous_spacecraft IS NOT NULL
                                 AND previous_spacecraft IS DISTINCT FROM spacecraft) AS source_changes
        FROM samples GROUP BY metric, bucket ORDER BY bucket, metric
    """), {'metrics': metrics, 'seconds': seconds, 'start': left, 'end': right})).mappings().all()
    saved = {(row['metric'], row['bucket']): row for row in rows}
    series = {}
    for metric in metrics:
        points, gaps = [], []
        valid = invalid = missing = 0
        cursor = left
        while cursor < right:
            row = saved.get((metric, cursor))
            count = row['count'] if row else 0
            present = row['samples'] if row else 0
            valid += count
            invalid += present - count
            missing += seconds//60 - present
            if row and count:
                points.append(dict(observed_at=cursor, interval_end=cursor+timedelta(seconds=seconds),
                    received_at=row['received_at'], value=row['mean'], min=row['min'], max=row['max'],
                    count=count, expected_count=seconds//60, coverage_percent=round(100*count/(seconds//60), 2),
                    spacecraft=row['spacecraft'] or '', last_spacecraft=row['last_spacecraft'] or '',
                    source_changes=row['source_changes'], quality='unverified', provider_quality=None,
                    negative_count=row['negative_count'] if metric == 'bz' else None))
            if count < seconds//60:
                reason = 'partial' if present else 'missing'
                if gaps and gaps[-1]['to'] == cursor and gaps[-1]['reason'] == reason:
                    gaps[-1]['to'] = cursor+timedelta(seconds=seconds)
                    gaps[-1]['slots'] += seconds//60-count
                else:
                    gaps.append({'from': cursor, 'to': cursor+timedelta(seconds=seconds),
                                 'reason': reason, 'slots': seconds//60-count})
            cursor += timedelta(seconds=seconds)
        expected = valid+invalid+missing
        series[metric] = {**metadata(metric), 'resolution_seconds': seconds, 'aggregation': 'mean_min_max',
            'points': points, 'coverage': {'expected_slots': expected, 'usable_slots': valid,
            'missing_slots': missing, 'invalid_slots': invalid, 'percent': round(100*valid/expected, 2) if expected else None,
            'resolution_seconds': 60, 'evaluated_to': right, 'basis': 'sample_timestamps', 'gaps': gaps}}
    return {'from': start, 'to': end, 'evaluated_from': left, 'evaluated_to': right,
            'resolution_seconds': seconds,
            'selection': 'complete_UTC_buckets_starting_before_to_left_edge_may_extend_before_from',
            'gap_filling': 'none', 'series': series}
