import asyncio
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from xml.etree import ElementTree as ET

import pytest
from common.schemas.forecast_release import ForecastArtifact, ForecastRelease, PREDICTION_COLUMNS, PRODUCT_ARTIFACTS
from common.schemas.solar_indices import SolarIndices
from common.schemas.solar_wind_events import SolarWindEvent
from prophet.services import summary as charts
from prophet.config import SummarySettings

NOW = datetime(2026, 10, 30, 12, 30, tzinfo=UTC)
ISSUE = NOW.replace(minute=0)
LEVELS = {'v': [450, 500, 600, 700], 'n': [10, 20, 50, 80], 't': [100000, 200000, 500000, 800000],
          'kp': [4, 5, 6, 7], 'ap': [27, 48, 80, 112], 'dst': [-30, -50, -100, -150],
          'bt': [5, 10, 15, 20], 'bx': [5, 10, 15, 20], 'by': [5, 10, 15, 20],
          'bz': [-5, -10, -15, -20], 'f10_7': [75, 150, 190],
          's10': [65, 150, 215], 'm10': [72, 144, 167], 'y10': [81, 148, 165]}


def release(product, *, future_peak=None, peak_lead=12, speed=500., threshold_peak_lead=None):
    artifacts = []
    for name in PRODUCT_ARTIFACTS[product]:
        columns = ['issue_time', 'valid_time', 'lead_hours', *PREDICTION_COLUMNS[name]]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        leads = range(24, 121, 24) if product == 'solar-radiation' else range(1, 13)
        for lead in leads:
            median = {'plasma_speed_quantile': speed, 'plasma_density_quantile': 5.,
                      'plasma_temperature_quantile': 100000., 'ap_quantile': 40., 'dst_quantile': -60.}.get(name, 150.)
            values = [.8, .2, .1] if name.endswith('_threshold') else [median-1, median, median+1]
            if name.endswith('_threshold') and lead == threshold_peak_lead:
                values = [.99]*3
            if future_peak is not None and lead == peak_lead and not name.endswith('_threshold'):
                values = [future_peak]*3
            writer.writerow(dict(issue_time=ISSUE.isoformat(), valid_time=(ISSUE+timedelta(hours=lead)).isoformat(),
                lead_hours=lead, **dict(zip(PREDICTION_COLUMNS[name], values))))
        content = buffer.getvalue()
        artifacts.append(ForecastArtifact(name=name, csv_text=content,
            sha256=hashlib.sha256(content.encode()).hexdigest(), row_count=len(leads), columns=columns, model_info={}))
    return ForecastRelease(product=product, run_id=uuid4(), release_id=uuid4(),
                           issue_time=ISSUE, published_at=NOW, artifacts=artifacts)

def install_sources(monkeypatch, tmp_path):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(charts, 'datetime', Clock)
    monkeypatch.setattr(charts, 'get_config', lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(charts, 'load_config', lambda: SimpleNamespace(summary=SummarySettings(critical_levels=LEVELS)))
    monkeypatch.setattr(charts, 'read_sdo_images', Mock(return_value={}))
    monkeypatch.setattr(charts, 'read_event_forecast', AsyncMock(return_value=None))
    monkeypatch.setenv('S3_PUBLIC_BASE_URL', 'https://media.example')
    values = {'v': 455., 'n': 5., 't': 100000., 'bx': 1., 'by': 2., 'bz': -12., 'bt': 13., 'kp': 4.3, 'dst': -60., 'ap': 48., 'f10_7': 150.}
    current = {m: dict(time=NOW if m in ('v','n','t','bx','by','bz','bt') else ISSUE, value=v, quality=None) for m,v in values.items()}
    histories = {m: [dict(time=ISSUE-timedelta(hours=i), value=v, quality=None) for i in (9,6,3,2,1)] for m,v in values.items()}
    monkeypatch.setattr(charts, 'read_observations', Mock(return_value=(current, histories)))
    solar = SolarIndices(as_of=NOW, items=[dict(metric=metric, value=100., observed_at=ISSUE-timedelta(days=day))
        for metric in ('f10_7', 's10', 'm10', 'y10') for day in (3,2,1,0)])
    monkeypatch.setattr(charts, 'read_solar_indices', AsyncMock(return_value=solar))
    releases = {p: release(p) for p in charts.PRODUCTS}
    monkeypatch.setattr(charts, 'read_release', AsyncMock(side_effect=releases.__getitem__))
    return current, histories, releases


def test_renders_quantiles_with_the_current_observation(monkeypatch, tmp_path):
    install_sources(monkeypatch, tmp_path)
    render = charts.render_image
    calls = []
    def capture(before, current, after, levels):
        calls.append((before, current, after, levels))
        return render(before, current, after, levels)
    monkeypatch.setattr(charts, 'render_image', capture)
    data, images = asyncio.run(charts.build_summary())
    assert len(images) == 12
    speed = next(call for call in calls if call[3] == LEVELS['v'])
    assert speed[1]['time'] == NOW and speed[1]['value'] == 455.
    assert [p['value'] for p in speed[2]] == [500.]*3
    kp = next(call for call in calls if call[3] == LEVELS['kp'])
    assert [p['time'] for p in kp[2]] == [ISSUE+timedelta(hours=i) for i in (3,6)]
    for metric in ('f10_7','s10','m10','y10'):
        before, current, after, _ = next(call for call in calls if call[3] == LEVELS[metric])
        assert [p['time'] for p in before] == [ISSUE-timedelta(days=i) for i in (3,2,1)]
        assert current['time'] == ISSUE
        assert [p['time'] for p in after] == [ISSUE+timedelta(days=i) for i in (1,2,3)]
    assert ET.fromstring(next(iter(images.values()))).tag.endswith('svg')


@pytest.mark.parametrize('missing', ['history', 'current', 'flagged'])
def test_missing_observations_omit_chart(monkeypatch, tmp_path, missing):
    current, histories, _ = install_sources(monkeypatch, tmp_path)
    if missing == 'history': histories['v'] = []
    elif missing == 'current': current.pop('v')
    else: current['v']['quality'] = 'flagged'
    data, images = asyncio.run(charts.build_summary())
    assert 'v' not in images
    assert data['observations']['plasma']['items'][0]['risk_level'] == 2


@pytest.mark.parametrize('fail_upload', [False,True])
def test_uploads_svg_and_atomically_saves_local_summary(monkeypatch, tmp_path, fail_upload):
    install_sources(monkeypatch, tmp_path)
    data, images = asyncio.run(charts.build_summary())
    monkeypatch.setattr(charts, 'build_summary', AsyncMock(return_value=(data, images)))
    uploaded = []
    def upload(content,bucket,prefix,name,**kwargs):
        if fail_upload and len(uploaded) == 1: raise OSError('upload failed')
        uploaded.append((content,bucket,prefix,name,kwargs))
        return prefix+'/'+name
    monkeypatch.setattr(charts.datamanager, 'upload_file', upload)
    if fail_upload:
        path = tmp_path/'prophet/summary/latest.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"previous": true}')
        with pytest.raises(OSError): asyncio.run(charts.publish_summary())
        assert len(uploaded) == 1 and uploaded[0][3].endswith('.svg')
        assert charts.read_summary() == {'previous': True}
    else:
        asyncio.run(charts.publish_summary())
        assert len(uploaded) == len(images)
        stored = charts.read_summary()
        speed = stored['observations']['plasma']['items'][0]
        assert speed['risk_level'] == 2
        assert speed['image_url'].startswith('https://media.example/forecasts/summary/images/v-')
        assert all(row[-1]['content_type']=='image/svg+xml' for row in uploaded)


def test_reads_prepared_summary_from_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(charts, 'get_config', lambda: SimpleNamespace(data_root=tmp_path))
    path = tmp_path/'prophet/summary/latest.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"observations": {"ready": true}}')
    assert charts.read_summary() == {'observations': {'ready': True}}


def test_missing_summary_is_not_generated_on_read(monkeypatch, tmp_path):
    monkeypatch.setattr(charts, 'get_config', lambda: SimpleNamespace(data_root=tmp_path))
    with pytest.raises(FileNotFoundError):
        charts.read_summary()


@pytest.mark.parametrize('probabilities,expected', [
    ([.1, .05, .01], 3.), ([.8, .2, .1], 4.5), ([.9, .8, .1], 5.5), ([.9, .8, .7], 7.),
    ([.2, .8, .1], 3.), ([.5, .5, .5], 3.),
])
def test_modal_threshold_interval(probabilities, expected):
    assert charts.threshold_value(probabilities, (4, 5, 6)) == expected


def test_numeric_clio_contract_preserves_metric_times_and_gaps(monkeypatch, tmp_path):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    responses = [
        [dict(time=(NOW-timedelta(minutes=2)).isoformat(), v=455., n=None),
         dict(time=(NOW-timedelta(hours=1)).isoformat(), v=None, n=5.),
         dict(time=(NOW+timedelta(minutes=1)).isoformat(), v=999.)],
        [dict(time=(ISSUE-timedelta(hours=1)).isoformat(), v=450., n=None, kp=2.)],
        [dict(time=(ISSUE-timedelta(hours=3)).isoformat(), kp=4., dst=-20.)],
    ]
    requests = []
    def handle(request):
        requests.append(request)
        return charts.httpx.Response(200, json={'success': True, 'data': {'points': responses[len(requests)-1]}})
    client = charts.httpx.Client(transport=charts.httpx.MockTransport(handle))
    monkeypatch.setattr(charts.httpx, 'Client', lambda **kwargs: client)
    current, histories = charts.read_observations(NOW)
    assert current['v']['time'] == NOW-timedelta(minutes=2) and current['v']['value'] == 455.
    assert current['n']['time'] == NOW-timedelta(hours=1)
    assert histories['v'][0]['value'] == 450. and 'n' not in histories
    assert histories['kp'][0]['value'] == 4.
    assert [r.url.path for r in requests] == ['/internal/v1/observations/latest',
        '/internal/v1/observations/history', '/internal/v1/observations/history']
    assert requests[1].url.params['interval'] == '1h'
    assert requests[2].url.params['interval'] == '1m'
    assert all(r.headers['Authorization'] == 'Bearer test-token' for r in requests)


def test_http_returns_prepared_summary_and_never_generates(monkeypatch, tmp_path):
    from prophet.main import app
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    data = {'observations': {'plasma': {'items': [{'risk_level': 2, 'image_url': 'https://media.example/v.svg'}]}}}
    read = Mock(return_value=data)
    monkeypatch.setattr(charts, 'read_summary', read)
    monkeypatch.setattr(charts, 'build_summary', Mock(side_effect=AssertionError('unexpected generation')))
    async def verify():
        async with charts.httpx.AsyncClient(transport=charts.httpx.ASGITransport(app=app), base_url='http://prophet') as client:
            assert (await client.get('/internal/v1/summary')).status_code == 401
            read.assert_not_called()
            headers = {'Authorization': 'Bearer test-token'}
            response = await client.get('/internal/v1/summary', headers=headers)
            assert response.status_code == 200
            assert response.json() == data
            read.side_effect = FileNotFoundError('private storage details')
            response = await client.get('/internal/v1/summary', headers=headers)
            assert response.status_code == 503 and 'private storage' not in response.text
    asyncio.run(verify())


def test_cme_age_filter_and_exact_public_fields(monkeypatch, tmp_path):
    install_sources(monkeypatch, tmp_path)
    def event(name, age, *, observed=True, kind='cme'):
        return SolarWindEvent(event_id=name, kind=kind, status='predicted_hit', source='DONKI/WSA-ENLIL',
            source_url='https://example.test/event', available_at=NOW if observed else NOW-age,
            observed_at=NOW-age if observed else None,
            arrival_target='Earth', arrival_at=NOW+timedelta(hours=2), arrival_window_start=NOW+timedelta(hours=1),
            arrival_window_end=NOW+timedelta(hours=3))
    events = [event('recent', timedelta(days=1)), event('boundary', timedelta(days=7)),
        event('old', timedelta(days=7, seconds=1)), event('updated-old', timedelta(days=8)),
        event('unknown-time', timedelta(days=2), observed=False),
        event('old-unknown-time', timedelta(days=8), observed=False), event('hss', timedelta(days=1), kind='hss')]
    monkeypatch.setattr(charts, 'read_event_forecast', AsyncMock(return_value=SimpleNamespace(events=events)))
    result, _ = asyncio.run(charts.build_summary())
    assert {event['event_id'] for event in result['cme_events']} == {'recent', 'boundary', 'unknown-time'}
    fields = {'event_id', 'source_url', 'observed_at', 'arrival_window_start', 'arrival_window_end'}
    assert all(set(event) == fields for event in result['cme_events'])


@pytest.mark.parametrize('changes,expected', [
    ({}, True),
    ({'glancing_blow': True, 'minor_impact': True}, True),
    ({'observed_at': None, 'event_id': 'enlil:multi-cme'}, True),
    ({'status': 'detected'}, False),
    ({'status': 'no_predicted_hit', 'arrival_at': None, 'arrival_target': None}, False),
    ({'arrival_at': None}, False),
    ({'arrival_target': None}, False),
    ({'available_at': NOW+timedelta(seconds=1)}, False),
    ({'status': 'prediction_elapsed', 'arrival_at': NOW-timedelta(hours=12)}, True),
    ({'status': 'prediction_elapsed', 'arrival_at': NOW-timedelta(hours=24)}, True),
    ({'status': 'prediction_elapsed', 'arrival_at': NOW-timedelta(hours=24, seconds=1)}, False),
    ({'status': 'prediction_elapsed', 'arrival_at': NOW-timedelta(hours=30),
      'arrival_window_start': NOW-timedelta(hours=32), 'arrival_window_end': NOW-timedelta(hours=12)}, True),
    ({'status': 'prediction_elapsed', 'arrival_at': NOW-timedelta(hours=30),
      'arrival_window_start': NOW-timedelta(hours=32), 'arrival_window_end': NOW-timedelta(hours=24, seconds=1)}, False),
])
def test_summary_keeps_only_relevant_earth_arrivals(changes, expected):
    values = dict(event_id='cme', kind='cme', status='predicted_hit', source='DONKI/WSA-ENLIL',
        available_at=NOW, observed_at=NOW-timedelta(days=2),
        arrival_at=NOW+timedelta(hours=12), arrival_target='Earth')
    event = SolarWindEvent(**{**values, **changes})
    assert bool(charts.cme_events(SimpleNamespace(events=[event]), NOW)) is expected


def test_cme_grace_period_is_configurable():
    event = SolarWindEvent(event_id='cme', kind='cme', status='prediction_elapsed', source='DONKI/WSA-ENLIL',
        available_at=NOW, observed_at=NOW-timedelta(days=2),
        arrival_at=NOW-timedelta(hours=6), arrival_target='Earth')
    events = SimpleNamespace(events=[event])
    assert charts.cme_events(events, NOW, grace_hours=6)
    assert not charts.cme_events(events, NOW, grace_hours=5)


@pytest.mark.parametrize('value,expected', [(449, 0), (450, 1), (500, 2), (600, 3), (700, 4), (-700, 4)])
def test_risk_levels_use_absolute_values_and_all_five_colors(value, expected):
    assert charts.risk_level({'value': value}, [], LEVELS['v']) == expected
    assert charts.risk_level(None, [value], LEVELS['v']) == expected


@pytest.mark.parametrize('configured', [
    {'cme_grace_hours': -1}, {'critical_levels': {'kp': []}}, {'critical_levels': {'kp': [4, 6, 5]}},
])
def test_invalid_settings(configured):
    with pytest.raises(ValueError):
        SummarySettings(**configured)


@pytest.mark.parametrize('missing', ['none', 'current'])
@pytest.mark.parametrize('peak_lead,expected', [(1, 4), (2, 4), (3, 4), (4, 0), (12, 0)])
def test_speed_risk_uses_only_current_and_three_hourly_forecast_points(monkeypatch, tmp_path, missing, peak_lead, expected):
    data, _, forecasts = install_sources(monkeypatch, tmp_path)
    data['v']['value'] = 325.
    forecasts['solar-wind-speed'] = release('solar-wind-speed', speed=350., future_peak=750., peak_lead=peak_lead)
    if missing == 'current':
        data.pop('v')
    result, _ = asyncio.run(charts.build_summary())
    speed = result['observations']['plasma']['items'][0]
    assert speed['risk_level'] == expected
    assert result['observations']['plasma']['risk_level'] == max(expected, 1)


@pytest.mark.parametrize('peak_lead,included', [(1, False), (2, False), (3, True), (6, True), (9, False), (12, False)])
def test_kp_ap_risk_uses_only_current_and_two_three_hour_forecast_points(monkeypatch, tmp_path, peak_lead, included):
    _, _, forecasts = install_sources(monkeypatch, tmp_path)
    forecasts['geomagnetic-activity'] = release('geomagnetic-activity', future_peak=150.,
                                               peak_lead=peak_lead, threshold_peak_lead=peak_lead)
    result, _ = asyncio.run(charts.build_summary())
    kp, ap, _ = result['observations']['geo']['items']
    assert kp['risk_level'] == (4 if included else 1)
    assert ap['risk_level'] == (4 if included else 2)



@pytest.mark.parametrize('state', ['ready', 'missing', 'invalid', 'unavailable'])
def test_summary_sdo_links_come_from_clio(monkeypatch, state):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    image = dict(channel='193', asset_id='2026103012_193_'+'a'*64,
        captured_at=NOW.isoformat(), urls={'webp.512': 'https://media.example/193.webp'})
    data = {'images': [image]} if state == 'ready' else {'images': []} if state == 'missing' else {'images': [{}]}
    def handle(request):
        assert request.url.path == '/internal/v1/observations/sdo'
        assert request.headers['Authorization'] == 'Bearer test-token'
        return charts.httpx.Response(503 if state == 'unavailable' else 200, json={'success': True, 'data': data})
    client = charts.httpx.Client(transport=charts.httpx.MockTransport(handle))
    monkeypatch.setattr(charts.httpx, 'Client', lambda **kwargs: client)
    result = charts.read_sdo_images()
    expected = {'193': {'asset_id': image['asset_id'], 'urls': image['urls']}} if state == 'ready' else {}
    assert result == expected
