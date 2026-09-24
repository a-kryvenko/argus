"""Read-only comparison of saved aggregates against retained raw observations."""
from datetime import timedelta
import math

from sqlalchemy import select, text

from argus_clio.db.models import SolarWindObservation, SolarWindAggregate, SolarWindAggregatePending
from argus_clio.services.solar_wind.aggregation import VERSION, summarize


def differences(expected, actual, path=''):
    """Report differing fields, allowing only floating-point rounding noise."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            name = f'{path}.{key}' if path else key
            if key not in expected or key not in actual:
                result.append(name)
            else:
                result.extend(differences(expected[key], actual[key], name))
        return result
    if type(expected) is float and type(actual) in (int, float):
        equal = math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
    else:
        equal = expected == actual
    return [] if equal else [path]


async def audit(session, start, end, now, retention_days=90, detail_limit=200):
    """Use a fresh session: every read belongs to one consistent read-only snapshot."""
    if (start >= end or end > now or end-start > timedelta(days=31)
            or any(value.minute or value.second or value.microsecond for value in (start, end))):
        raise ValueError('Audit requires closed UTC hours and a range of at most 31 days')
    if retention_days < 1 or detail_limit < 1:
        raise ValueError('Retention and detail limits must be positive')
    await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
    await session.execute(text("SET LOCAL statement_timeout = '60s'"))
    cutoff = now-timedelta(days=retention_days)
    storage = dict((await session.execute(text("""
        SELECT count(*) AS total_rows,
               count(*) FILTER (WHERE observed_at < :cutoff) AS older_rows,
               count(*) FILTER (WHERE observed_at < :cutoff AND active) AS older_active_rows,
               coalesce(sum(pg_column_size(r)) FILTER (WHERE observed_at < :cutoff), 0) AS older_payload_bytes,
               min(observed_at) AS first_observation,
               max(observed_at) AS last_observation,
               pg_total_relation_size('solar_wind_observation'::regclass) AS allocated_bytes
        FROM solar_wind_observation r
    """), {'cutoff': cutoff})).mappings().one())
    total = storage['total_rows']
    storage['estimated_older_allocation_bytes'] = round(storage['allocated_bytes']*storage['older_rows']/total) if total else 0
    storage['cutoff'] = cutoff
    storage['retention_days'] = retention_days
    # Include aggregate-only hours: absence of retained raw data is not proof of
    # a correct empty aggregate. Do not certify such hours for future deletion.
    hours = list((await session.execute(text("""
        SELECT kind, date_trunc('hour', observed_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' AS hour
        FROM solar_wind_observation WHERE observed_at >= :start AND observed_at < :end
        UNION
        SELECT kind, date_trunc('hour', bucket_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' AS hour
        FROM solar_wind_aggregate WHERE bucket_start >= :start AND bucket_start < :end
        UNION
        SELECT kind, hour FROM solar_wind_aggregate_pending WHERE hour >= :start AND hour < :end
        ORDER BY hour, kind
    """), {'start': start, 'end': end})).mappings())
    pending = {(r.kind, r.hour) for r in (await session.execute(select(SolarWindAggregatePending).where(
        SolarWindAggregatePending.hour >= start, SolarWindAggregatePending.hour < end))).scalars()}
    counts = dict(source_hours=len(hours), checked_buckets=0, matched_buckets=0,
                  missing_buckets=0, mismatched_buckets=0, unverifiable_buckets=0,
                  pending_source_hours=len(pending))
    issues = []
    issue_count = 0

    def issue(kind, hour, reason, **extra):
        nonlocal issue_count
        issue_count += 1
        if len(issues) < detail_limit:
            issues.append({'kind': kind, 'hour': hour, 'reason': reason, **extra})

    for item in hours:
        kind, hour = item['kind'], item['hour']
        if (kind, hour) in pending:
            issue(kind, hour, 'recalculation_pending')
        raw = list((await session.execute(select(SolarWindObservation).where(
            SolarWindObservation.kind == kind, SolarWindObservation.observed_at >= hour,
            SolarWindObservation.observed_at < hour+timedelta(hours=1)))).scalars())
        saved = {(r.resolution_seconds, r.bucket_start): r for r in (await session.execute(
            select(SolarWindAggregate).where(SolarWindAggregate.kind == kind,
                SolarWindAggregate.bucket_start >= hour,
                SolarWindAggregate.bucket_start < hour+timedelta(hours=1)))).scalars()}
        for seconds in (300, 3600):
            for offset in range(0, 3600, seconds):
                bucket = hour+timedelta(seconds=offset)
                record = saved.get((seconds, bucket))
                extra = {'bucket_start': bucket, 'resolution_seconds': seconds}
                if not raw:
                    counts['unverifiable_buckets'] += 1
                    issue(kind, hour, 'raw_unavailable', **extra)
                    continue
                counts['checked_buckets'] += 1
                if record is None:
                    counts['missing_buckets'] += 1
                    issue(kind, hour, 'aggregate_missing', **extra)
                    continue
                expected = summarize(raw, kind, bucket, seconds, now)
                fields = differences(expected, record.statistics)
                if record.version != VERSION:
                    fields.append('version')
                if fields:
                    counts['mismatched_buckets'] += 1
                    issue(kind, hour, 'aggregate_mismatch', fields=fields, **extra)
                else:
                    counts['matched_buckets'] += 1
    status = 'issues_found' if issue_count else 'ok' if counts['checked_buckets'] else 'no_data'
    return {'generated_at': now, 'from': start, 'to': end, 'aggregation_version': VERSION,
            'status': status, 'counts': counts, 'issue_count': issue_count,
            'issues': issues, 'details_truncated': issue_count > len(issues), 'storage': storage,
            'scope': 'occupied_source_hours_in_range', 'read_only': True,
            'notes': [
                'Comparison uses retained raw rows and the current aggregation rule; it does not prove upstream completeness.',
                'Storage totals cover the entire raw table; comparison covers only the requested range.',
                'Older allocation is a proportional estimate including indexes, not disk space guaranteed to be released.',
                'This snapshot does not authorize deletion or persist a cleanup eligibility flag.',
            ]}
