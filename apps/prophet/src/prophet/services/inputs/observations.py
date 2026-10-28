"""Assemble model context using consumer-independent observation resources."""
from datetime import UTC, datetime, timedelta

from common.schemas.observation import Observation
from forecast.observation_inputs import ObservationInputs

HISTORY_DAYS = 60


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

    # Clio intervals are half-open; retain the inclusive model cutoff.
    end = (as_of + timedelta(microseconds=1)).isoformat()
    start = (as_of - timedelta(days=HISTORY_DAYS)).isoformat()
    observations = Observation.model_validate(read('history', **{'from': start, 'to': end, 'limit': 24*HISTORY_DAYS+1}))

    def measurements(metrics, since, **params):
        data = read('measurements', metrics=metrics, start=since, end=end, **params)
        if data.get('schema_version') != 1:
            raise ValueError('Unsupported measurement schema version')
        return data['items']

    indices = measurements('f10_7,dst,ap', (as_of - timedelta(days=88)).isoformat())
    # Filter legacy stored-history missing codes before hourly aggregation.
    speed = measurements('v', start, aggregation='hourly', min_value=0, max_value=9999)
    density = measurements('n', start, aggregation='hourly', min_value=0, exclude_value=999.9)
    files = []
    for kind, days in (('gong', 2), ('goes', 90), ('aia', 45)):
        files.extend(read('files', kind=kind, start=(as_of-timedelta(days=days)).isoformat(),
                          end=end, as_of=as_of.isoformat()))
    issue = as_of.replace(minute=0, second=0, microsecond=0)
    hourly = read('solar-wind/history', metrics='bx,by,bz,v,n,t', resolution='1h',
                  **{'from': (issue-timedelta(hours=168)).isoformat(), 'to': issue.isoformat()})
    return ObservationInputs(
        as_of=as_of, read_at=read_at, observations=observations, measurements=indices, files=files,
        speed_observations=[{'issue_time': row['observed_at'], 'v': row['value']} for row in speed],
        density_observations=[{'issue_time': row['observed_at'], 'n': row['value']} for row in density],
        solar_wind_hourly=hourly,
    )
