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
                # Clio filters source receipts by this cutoff. It is an availability
                # bound for this snapshot, not a claimed original source receipt.
                received_at=as_of, quality='unverified', value=sum(numbers)/len(numbers),
                count=len(numbers), expected_count=60, min=min(numbers), max=max(numbers)))
        series[metric] = dict(points=rows, unit={'v': 'km/s', 'n': 'cm⁻³', 't': 'K'}.get(metric, 'nT'),
            location='L1', time_basis='measurement', propagated=False, aggregation='mean_min_max',
            coordinate_system='GSM' if metric in ('bx', 'by', 'bz') else None)
    return dict(resolution_seconds=3600, gap_filling='none', series=series)


def read_observations(client, url, token, as_of, *, request=None):
    request = request or client.get
    headers = {'Authorization': f'Bearer {token}'}
    read_at = datetime.now(UTC)

    def read(resource, **params):
        response = request(url.rstrip('/') + '/internal/v1/observations/' + resource,
                           params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if payload.get('success') is not True or payload.get('data') is None:
            raise ValueError('Observation service returned an unsuccessful response')
        return payload['data']

    end = (as_of + timedelta(microseconds=1)).isoformat()
    start = (as_of - timedelta(days=HISTORY_DAYS)).isoformat()
    hourly = read('metrics', metrics=','.join((*CORE, *MEANS)), interval='1h',
                  as_of=as_of.isoformat(), **{'from': start, 'to': end})
    if hourly.get('interval') != '1h' or not isinstance(hourly.get('points'), list):
        raise ValueError('Expected Clio hourly metric history')
    rows = hourly['points']
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
    issue = as_of.replace(minute=0, second=0, microsecond=0)
    native = read('metrics', metrics=','.join(WIND), interval='1m', as_of=as_of.isoformat(),
                  **{'from': (issue-timedelta(hours=168)).isoformat(), 'to': issue.isoformat()})
    if native.get('interval') != '1m' or not isinstance(native.get('points'), list):
        raise ValueError('Expected Clio minute metric history')
    files = []
    for kind, days in (('gong', 2), ('goes', 35), ('aia', 45)):
        files.extend(read('files', kind=kind, start=(as_of-timedelta(days=days)).isoformat(),
                          end=end, as_of=as_of.isoformat()))
    return ObservationInputs(as_of=as_of, read_at=read_at, observations=observations,
        measurements=indices, speed_observations=speed, density_observations=density,
        files=files, solar_wind_hourly=native_hourly(native['points'], issue, as_of))
