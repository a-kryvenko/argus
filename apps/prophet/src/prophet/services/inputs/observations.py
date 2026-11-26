"""Forecast-owned history windows over Clio's compact numeric and file resources."""
from datetime import UTC, datetime, timedelta

from common.observationcatalog import METRICS
from common.schemas.observation import Observation
from forecast.observation_inputs import ObservationInputs

HISTORY_DAYS = 60
CORE = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', 'ap', 'f10_7')
MEANS = ('f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean')
WIND = ('bx', 'by', 'bz', 'v', 'n', 't')
INTEGER_MODEL_METRICS = ('kp', 'dst', 'ap', 'f10_7')


def native_hourly(aggregates, as_of):
    """Adapt Clio's closed L1 windows to the forecast feature contract."""
    series = {}
    for metric in WIND:
        record = aggregates.get(metric, {})
        if (record.get('time_basis'), record.get('aggregation'), record.get('resolution_seconds')) != ('l1', 'mean', 3600):
            raise ValueError(f'Expected hourly L1 aggregates for {metric}')
        rows = []
        for point in record['points']:
            # Features are causal and cannot use a window with unknown receipt.
            if (point.get('received_at') is None
                    or datetime.fromisoformat(point['received_at']) > as_of):
                continue
            hour = datetime.fromisoformat(point['observed_at'])
            rows.append({**point, 'interval_end': hour + timedelta(hours=1),
                         'quality': point.get('quality') or 'unverified'})
        series[metric] = dict(points=rows, unit=METRICS[metric]['unit'],
            location='L1', time_basis='measurement', propagated=False, aggregation='mean_min_max',
            coordinate_system=METRICS[metric].get('frame'))
    return dict(resolution_seconds=3600, gap_filling='none', series=series)


def read_resource(request, url, token, path, **params):
    response = request(url.rstrip('/') + path, params=params, headers={'Authorization': f'Bearer {token}'})
    response.raise_for_status()
    payload = response.json()
    if payload.get('success') is not True or payload.get('data') is None:
        raise ValueError('Observation service returned an unsuccessful response')
    return payload['data']


def read_series(request, url, token, metrics, **params):
    data = read_resource(request, url, token, '/internal/v1/observations/series',
                         metrics=','.join(metrics), **params)
    if not isinstance(data, dict) or any(
        not isinstance(series, dict) or not isinstance(series.get('points'), list) for series in data.values()
    ):
        raise ValueError('Expected Clio observation series')
    return data


def read_model_inputs(request, url, token, start, end):
    data = read_resource(request, url, token, '/internal/v1/observations/model-inputs',
                         metrics=','.join((*CORE, *MEANS)),
                         **{'from': start.isoformat(), 'to': end.isoformat()})
    if not isinstance(data, dict) or data.get('interval') != '1h' or not isinstance(data.get('points'), list):
        raise ValueError('Expected Clio hourly model inputs')
    return data['points']


def read_observations(client, url, token, as_of, *, request=None):
    read_at = datetime.now(UTC)
    rows = read_model_inputs(request or client.get, url, token,
                             as_of - timedelta(days=HISTORY_DAYS), as_of + timedelta(microseconds=1))
    observations = Observation.model_validate({'points': [
        dict(issue_time=row['time'], **{metric: int(row[metric]) if metric in INTEGER_MODEL_METRICS else row[metric]
                                      for metric in CORE})
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
    raw = read_series(client.get, url, token, WIND, interval='1h', time_basis='l1',
                      **{'from': (issue - timedelta(hours=168)).isoformat(), 'to': issue.isoformat()})
    return native_hourly(raw, as_of)


def apply_solar_measurements(inputs, calibrated):
    inputs.measurements = [row for row in inputs.measurements if row.metric not in ('s10', 'm10', 'y10')] + calibrated
    by_time = {}
    for row in calibrated:
        by_time.setdefault(row.observed_at, {})[row.metric] = row.value
    for point in inputs.observations.points:
        for metric in ('s10', 'm10', 'y10'):
            setattr(point, metric, by_time.get(point.issue_time, {}).get(metric))
