from datetime import UTC, datetime, timedelta

import httpx

from forecast.inputs import GeomagneticHistory
from prophet.services.inputs.observations import read_geomagnetic_history


def test_geomagnetic_history_uses_closed_l1_and_native_index_intervals():
    issue = datetime(2026, 11, 26, 12, tzinfo=UTC)
    as_of = issue + timedelta(minutes=8)
    history_time = issue - timedelta(days=27)
    requests = []

    def point(at, value, *, receipt=as_of, **details):
        return dict(observed_at=at.isoformat(), received_at=receipt.isoformat(), value=value, **details)

    def response(request):
        requests.append(request)
        if request.url.path.endswith('/model-inputs'):
            assert request.url.params['metrics'] == 'kp'
            assert request.url.params['from'] == (issue - timedelta(days=29)).isoformat()
            assert request.url.params['to'] == (issue - timedelta(days=24)).isoformat()
            rows = [
                dict(time=history_time.isoformat(), kp=3.67),
                dict(time=(history_time + timedelta(hours=1)).isoformat(), kp=4.33),
                dict(time=(issue - timedelta(hours=3)).isoformat(), kp=6.33),
            ]
            return httpx.Response(200, json={'success': True, 'data': {'interval': '1h', 'points': rows}})

        assert request.url.path == '/internal/v1/observations/series'
        assert request.url.params['interval'] == '1h'
        assert request.url.params['time_basis'] == 'l1'
        assert request.url.params['to'] == issue.isoformat()
        start = issue - timedelta(hours=1)
        wind = {
            'n': [point(start, 2.5, count=45), point(issue, 100, count=60)],
            'v': [point(start, 350, count=44)],
            'by': [point(start, 12, count=60, receipt=as_of + timedelta(seconds=1))],
            'bz': [point(start, -5, count=60, quality='flagged')],
        }
        data = {key: dict(resolution_seconds=3600, time_basis='l1', aggregation='mean', points=rows)
                for key, rows in wind.items()}
        data['kp'] = dict(resolution_seconds=10800, time_basis='source_interval_start', aggregation='source',
                          points=[point(issue - timedelta(hours=3), 4.67), point(issue, 7)])
        data['dst'] = dict(resolution_seconds=3600, time_basis='source_interval_start', aggregation='source',
                           points=[point(start, -11.5), point(issue, -40)])
        data['f10_7'] = dict(resolution_seconds=86400, time_basis='measurement', aggregation='source', points=[
            point(datetime(2026, 11, 25, 20, tzinfo=UTC), 113.7),
            point(datetime(2026, 11, 26, 11, tzinfo=UTC), 121.2),
        ])
        return httpx.Response(200, json={'success': True, 'data': data})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        result = read_geomagnetic_history(client, 'http://clio', 'token', as_of)

    assert len(requests) == 2
    assert [(point.time, point.kp) for point in result.raw] == [(history_time, 3.67), (issue-timedelta(hours=3), 4.67)]
    prior = {row.time: row for row in result.prior}
    assert prior[issue].model_dump(exclude_none=True) == {'time': issue, 'kp': 4.67, 'dst': -11.5, 'f10_7': 113.7, 'n': 2.5}
    assert prior[issue - timedelta(hours=1)].kp is None
    assert max(prior) == issue
    assert GeomagneticHistory.model_validate_json(result.model_dump_json()) == result
