"""Diagnostic evidence only: no inferred scientific freshness thresholds."""
import math
from datetime import UTC, datetime

from common.schemas.forecast_inputs import DENSITY_METRICS
from common.schemas.forecast_status import ForecastStatus
from common.schemas.forecast_release import PRODUCT_ARTIFACTS


NORMALIZED_FIELDS = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', 'ap', 'f10_7', 's10', 'm10', 'y10')


def input_diagnostics(inputs):
    as_of = inputs.as_of.astimezone(UTC)
    timestamps = []
    invalid_timestamps = 0
    invalid_values = {name: 0 for name in NORMALIZED_FIELDS}
    for point in inputs.observations.points:
        timestamp = point.issue_time
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            invalid_timestamps += 1
        else:
            timestamps.append(timestamp.astimezone(UTC))
        for name in NORMALIZED_FIELDS:
            value = getattr(point, name)
            if value is None or not math.isfinite(value):
                invalid_values[name] += 1
    latest = max(timestamps, default=None)
    unique = sorted(set(timestamps))
    sources = {}
    for metric in DENSITY_METRICS:
        times = [row.observed_at.astimezone(UTC) for row in inputs.measurements if row.metric == metric]
        last = max(times, default=None)
        sources[metric] = {
            'count': len(times), 'latest_at': last.isoformat() if last else None,
            'age_hours': (as_of - last).total_seconds() / 3600 if last else None,
            'future_count': sum(time > as_of for time in times),
        }
    issues = []
    if not timestamps:
        issues.append('no_aware_normalized_timestamps')
    if invalid_timestamps:
        issues.append('naive_normalized_timestamps')
    if len(unique) != len(timestamps):
        issues.append('duplicate_normalized_timestamps')
    if any(time > as_of for time in timestamps):
        issues.append('future_normalized_timestamps')
    if any(invalid_values[name] for name in NORMALIZED_FIELDS[:10]):
        issues.append('missing_or_nonfinite_required_values')
    if any(row['future_count'] for row in sources.values()):
        issues.append('future_source_measurements')
    return {
        'version': 1, 'mode': 'observe', 'as_of': as_of.isoformat(), 'read_at': inputs.read_at.isoformat(),
        'thresholds_configured': False, 'issues': issues,
        'normalized': {
            'count': len(inputs.observations.points),
            'first_at': min(timestamps).isoformat() if timestamps else None,
            'latest_at': latest.isoformat() if latest else None,
            'age_hours': (as_of - latest).total_seconds() / 3600 if latest else None,
            'naive_timestamp_count': invalid_timestamps,
            'duplicate_timestamp_count': len(timestamps) - len(unique),
            'max_gap_hours': max(((b - a).total_seconds() / 3600 for a, b in zip(unique, unique[1:])), default=None),
            'invalid_value_counts': invalid_values,
            'source_freshness_known': False,
        },
        'density_sources': sources,
    }


def classify_freshness(issue_time, now, max_age_hours=None):
    if issue_time is None:
        return 'unavailable'
    age = (now - issue_time).total_seconds() / 3600
    if age < 0:
        return 'future_issue_time'
    if max_age_hours is None:
        return 'unconfigured'
    return 'stale' if age > max_age_hours else 'within_age_limit'


def product_status(product, *, now=None):
    """Read metadata without decompressing forecast payloads or running models."""
    from psycopg.rows import dict_row
    from common.config import get_config
    from argus_prophet.db.session import connect
    if product not in PRODUCT_ARTIFACTS:
        raise ValueError('Unknown forecast product')
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Assessment time must be timezone-aware')
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor.execute('''SELECT r.id AS release_id,r.run_id,r.issue_time,r.published_at,
            f.started_at,f.provenance->'input_diagnostics' AS input_diagnostics
            FROM prophet.current_forecast c JOIN prophet.forecast_release r ON r.id=c.release_id
            JOIN prophet.forecast_run f ON f.id=r.run_id WHERE c.product=%s''', (product,))
        current = cursor.fetchone()
        from argus_prophet.products import attempt_selections
        cursor.execute("""SELECT id AS run_id,status,started_at,finished_at,error,
            provenance->'input_diagnostics' AS input_diagnostics FROM prophet.forecast_run
            WHERE product=ANY(%s) ORDER BY started_at DESC,id DESC LIMIT 1""", (attempt_selections(product),))
        attempt = cursor.fetchone()
        artifacts = []
        if attempt:
            cursor.execute("""SELECT name,status,error FROM prophet.forecast_artifact
                WHERE run_id=%s AND name=ANY(%s) ORDER BY name""",
                (attempt['run_id'], list(PRODUCT_ARTIFACTS[product])))
            artifacts = cursor.fetchall()
    max_age = None
    if product == 'atmospheric-density':
        max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    age = (now - current['issue_time']).total_seconds() / 3600 if current else None
    return ForecastStatus(**{
        'contract_version': 1, 'product': product, 'assessed_at': now,
        'mode': 'observe', 'current_release': current,
        'release_age_hours': age,
        'existing_public_max_age_hours': max_age,
        'freshness': classify_freshness(current['issue_time'] if current else None, now, max_age),
        'latest_attempt': attempt, 'latest_attempt_artifacts': artifacts,
        'note': 'Diagnostic status does not establish model readiness or individual sensor freshness.',
    })
