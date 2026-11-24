"""Forecast-owned history windows over Clio's compact numeric and file resources."""
from datetime import UTC, datetime, timedelta
import math

from common.schemas.observation import Observation
from forecast.observation_inputs import ObservationInputs

HISTORY_DAYS = 60
CORE = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', 'ap', 'f10_7')
MEANS = ('f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean')
WIND = ('bx', 'by', 'bz', 'v', 'n', 't')


def native_hourly(points, issue, as_of):
    """Build model inputs from actual minute samples, retaining gaps and counts."""
    from collections import defaultdict
    grouped = {metric: defaultdict(dict) for metric in WIND}
    for row in points:
        time = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
        hour = time.replace(minute=0, second=0, microsecond=0)
        if hour + timedelta(hours=1) > issue:
            continue
        for metric in WIND:
            number = row.get(metric)
            if number is not None and math.isfinite(number):
                grouped[metric][hour][time.replace(second=0, microsecond=0)] = number
    series = {}
    for metric in WIND:
        rows = []
        for hour, samples in sorted(grouped[metric].items()):
            numbers = list(samples.values())
            rows.append(dict(observed_at=hour, interval_end=hour+timedelta(hours=1),
                # Snapshot time; the compact history has no per-sample receipt.
                received_at=as_of, quality='unverified', value=sum(numbers)/len(numbers),
                count=len(numbers), expected_count=60, min=min(numbers), max=max(numbers)))
        series[metric] = dict(points=rows, unit={'v': 'km/s', 'n': 'cm⁻³', 't': 'K'}.get(metric, 'nT'),
            location='L1', time_basis='measurement', propagated=False, aggregation='mean_min_max',
            coordinate_system='GSM' if metric in ('bx', 'by', 'bz') else None)
    return dict(resolution_seconds=3600, gap_filling='none', series=series)


def read_resource(request, url, token, path, **params):
    response = request(url.rstrip('/') + path, params=params, headers={'Authorization': f'Bearer {token}'})
    response.raise_for_status()
    payload = response.json()
    if payload.get('success') is not True or payload.get('data') is None:
        raise ValueError('Observation service returned an unsuccessful response')
    return payload['data']


def read_history(request, url, token, metrics, interval, start, end):
    data = read_resource(request, url, token, '/internal/v1/observations/history',
                         metrics=','.join(metrics), interval=interval,
                         **{'from': start.isoformat(), 'to': end.isoformat()})
    if not isinstance(data, dict) or data.get('interval') != interval or not isinstance(data.get('points'), list):
        raise ValueError(f'Expected Clio {interval} metric history')
    return data['points']


def read_observations(client, url, token, as_of, *, request=None):
    read_at = datetime.now(UTC)
    rows = read_history(request or client.get, url, token, (*CORE, *MEANS), '1h',
                        as_of - timedelta(days=HISTORY_DAYS), as_of + timedelta(microseconds=1))
    observations = Observation.model_validate({'points': [
        dict(issue_time=row['time'], **{metric: row[metric] for metric in CORE})
        for row in rows if all(row.get(metric) is not None for metric in CORE)]})
    indices = [dict(metric=metric, value=row[metric], observed_at=row['time'])
               for row in rows if datetime.fromisoformat(row['time'].replace('Z', '+00:00')) >= as_of-timedelta(days=35)
               for metric in ('f10_7', 'dst', 'ap', *MEANS) if row.get(metric) is not None]
    speed = [dict(issue_time=row['time'], v=row['v']) for row in rows
             if row.get('v') is not None and 0 <= row['v'] < 9999]
    density = [dict(issue_time=row['time'], n=row['n']) for row in rows
               if row.get('n') is not None and row['n'] >= 0 and row['n'] != 999.9]
    return ObservationInputs(as_of=as_of, read_at=read_at, observations=observations,
        measurements=indices, speed_observations=speed, density_observations=density)


def read_native_history(client, url, token, as_of):
    issue = as_of.replace(minute=0, second=0, microsecond=0)
    rows = read_history(client.get, url, token, WIND, '1m', issue-timedelta(hours=168), issue)
    return native_hourly(rows, issue, as_of)


def apply_solar_measurements(inputs, calibrated):
    inputs.measurements = [row for row in inputs.measurements if row.metric not in ('s10', 'm10', 'y10')] + calibrated
    by_time = {}
    for row in calibrated:
        by_time.setdefault(row.observed_at, {})[row.metric] = row.value
    for point in inputs.observations.points:
        for metric in ('s10', 'm10', 'y10'):
            setattr(point, metric, by_time.get(point.issue_time, {}).get(metric))
