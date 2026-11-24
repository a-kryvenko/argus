"""Diagnostic evidence only: no inferred scientific freshness thresholds."""
import math
from datetime import UTC

from forecast.observation_inputs import DENSITY_METRICS


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
    if not timestamps and inputs.solar_wind_events is None:
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
        'solar_wind_events': ({'poll_count': len(inputs.solar_wind_events.polls),
                              'read_at': inputs.solar_wind_events.read_at.isoformat()}
                             if inputs.solar_wind_events else None),
    }
