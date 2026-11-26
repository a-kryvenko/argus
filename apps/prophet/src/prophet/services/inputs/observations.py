"""Forecast-owned history windows over Clio's compact numeric and file resources."""
from datetime import UTC, datetime, timedelta
from math import isfinite

from pydantic import ValidationError

from common.observationcatalog import METRICS
from common.schemas.observation import Observation
from forecast.inputs import F107Measurement, GeomagneticHistory
from forecast.observation_inputs import ObservationInputs

HISTORY_DAYS = 60
CORE = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', 'ap', 'f10_7')
MEANS = ('f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean')
WIND = ('bx', 'by', 'bz', 'v', 'n', 't')
INTEGER_MODEL_METRICS = ('kp', 'dst', 'ap', 'f10_7')
GEOMAGNETIC_WIND = ('by', 'bz', 'v', 'n')
GEOMAGNETIC_INDICES = ('kp', 'dst', 'f10_7')
RECURRENCE_START = timedelta(days=29)
RECURRENCE_END = timedelta(days=24)


def native_hourly(aggregates, as_of, *, metrics=WIND):
    """Adapt Clio's closed L1 windows to the forecast feature contract."""
    series = {}
    for metric in metrics:
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


def read_model_inputs(request, url, token, start, end, *, metrics=(*CORE, *MEANS)):
    data = read_resource(request, url, token, '/internal/v1/observations/model-inputs',
                         metrics=','.join(metrics),
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


def _usable_value(point):
    value = point.get('value')
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
            and point.get('quality') in (None, '', 'good', 'unverified'))


def read_f107_history(client, url, token, as_of):
    """Read only native Penticton observations received by the input cutoff."""
    history = read_series(client.get, url, token, ('f10_7',), received_by=as_of.isoformat(),
                          **{'from': (as_of-timedelta(days=35)).isoformat(),
                             'to': (as_of+timedelta(microseconds=1)).isoformat()})
    series = history.get('f10_7', {})
    if (series.get('resolution_seconds'), series.get('time_basis'), series.get('aggregation')) != (86400, 'measurement', 'source'):
        raise ValueError('Expected native Clio F10.7 measurements')
    points = []
    for point in series['points']:
        if not isinstance(point, dict) or not _usable_value(point):
            continue
        try:
            observation = F107Measurement.model_validate(point)
        except ValidationError:
            continue
        if observation.observed_at <= observation.received_at <= as_of:
            points.append(observation)
    return points


def _hour(stamp):
    return stamp.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _kp_start(stamp):
    hour = _hour(stamp)
    return stamp == hour and hour.hour % 3 == 0


def read_geomagnetic_history(client, url, token, as_of):
    """Build the ordinal model's feature clock from closed, received observations."""
    issue = _hour(as_of)
    recent_start = issue - timedelta(hours=168)
    aggregate = read_series(client.get, url, token, (*GEOMAGNETIC_WIND, *GEOMAGNETIC_INDICES),
                            interval='1h', time_basis='l1',
                            **{'from': recent_start.isoformat(), 'to': issue.isoformat()})
    recurrence_start, recurrence_end = issue - RECURRENCE_START, issue - RECURRENCE_END
    recurrence = read_model_inputs(client.get, url, token, recurrence_start, recurrence_end, metrics=('kp',))

    raw = {}
    for row in recurrence:
        stamp = datetime.fromisoformat(row['time'])
        kp = row.get('kp')
        if recurrence_start <= stamp < recurrence_end and _kp_start(stamp) and kp is not None and 0 <= kp <= 9:
            raw[stamp] = kp

    prior = {}
    wind = native_hourly(aggregate, as_of, metrics=GEOMAGNETIC_WIND)
    for metric, series in wind['series'].items():
        for point in series['points']:
            start = datetime.fromisoformat(point['observed_at'])
            end = point['interval_end']
            if (recent_start <= start and start == _hour(start) and end <= issue
                    and 45 <= (point.get('count') or 0) <= 60 and _usable_value(point)):
                prior.setdefault(end, {})[metric] = point['value']

    for metric, (hours, basis, carry) in {
        'kp': (3, 'source_interval_start', 3),
        'dst': (1, 'source_interval_start', 1),
        'f10_7': (24, 'measurement', 25),
    }.items():
        series = aggregate.get(metric, {})
        contract = (series.get('resolution_seconds'), series.get('time_basis'), series.get('aggregation'))
        if contract != (hours * 3600, basis, 'source'):
            raise ValueError(f'Expected native Clio series for {metric}')
        ends = {}
        for point in series['points']:
            receipt = point.get('received_at')
            if not _usable_value(point) or receipt is None or datetime.fromisoformat(receipt) > as_of:
                continue
            stamp = datetime.fromisoformat(point['observed_at'])
            if (metric == 'kp' and not _kp_start(stamp)) or (metric == 'dst' and stamp != _hour(stamp)):
                continue
            start = stamp.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0) if metric == 'f10_7' else stamp
            end = start + timedelta(hours=hours)
            if end > issue or stamp > as_of:
                continue
            ends[end] = point['value']
            if metric == 'kp':
                raw[stamp] = point['value']
        for end, value in sorted(ends.items()):
            for offset in range(carry):
                clock = end + timedelta(hours=offset)
                if recent_start <= clock <= issue:
                    prior.setdefault(clock, {})[metric] = value

    return GeomagneticHistory(raw=[{'time': stamp, 'kp': kp} for stamp, kp in sorted(raw.items())],
                              prior=[{'time': stamp, **values} for stamp, values in sorted(prior.items())])


def apply_solar_measurements(inputs, calibrated):
    inputs.measurements = [row for row in inputs.measurements if row.metric not in ('s10', 'm10', 'y10')] + calibrated
    by_time = {}
    for row in calibrated:
        by_time.setdefault(row.observed_at, {})[row.metric] = row.value
    for point in inputs.observations.points:
        for metric in ('s10', 'm10', 'y10'):
            setattr(point, metric, by_time.get(point.issue_time, {}).get(metric))
