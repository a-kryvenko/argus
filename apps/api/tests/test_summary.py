import asyncio
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException
import httpx
import pytest

from common.schemas.forecast_release import ForecastArtifact, ForecastRelease, PREDICTION_COLUMNS, PRODUCT_ARTIFACTS
from common.schemas.solar_wind_events import SolarWindEvent
from common.schemas.sdo_browse import CHANNELS
from common.schemas.solar_indices import SolarIndices
from types import SimpleNamespace
from api.schemas.summary import SummarySettings
from api.services.errors import ArtifactNotReadyError
from api.services import summary
from api.routers import summary as routes

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


def point(time, value):
    return {'observed_at': time.isoformat(), 'value': value, 'quality': 'unverified'}


def sources():
    values = {'v': 455., 'n': 5., 't': 100000., 'bx': 1., 'by': 2., 'bz': -12., 'bt': 13.}
    wind = {'series': {m: {'latest': point(NOW, value)} for m, value in values.items()}}
    geo = {'series': {m: {'latest': point(ISSUE, value)} for m, value in [('kp', 4.3), ('dst', -60.)]}}
    additional = {'series': {'ap': point(ISSUE, 48.), 'f10_7': point(ISSUE, 150.),
                             's10': None, 'm10': None, 'y10': None}}
    wind_history = {'series': {m: {'points': [point(ISSUE-timedelta(hours=i), value) for i in (3, 2, 1)]}
                               for m, value in values.items()}}
    geo_history = {'series': {m: {'points': [point(ISSUE-timedelta(hours=i), value) for i in (9, 6, 3, 2, 1)]}
                              for m, value in [('kp', 4.), ('dst', -20.)]}}
    hourly = {'points': [dict(issue_time=(ISSUE-timedelta(hours=i)).isoformat(), ap=27., f10_7=120.)
                         for i in (9, 6, 3, 2, 1)]}
    result = dict(zip(('solar-wind/latest', 'geomagnetic/latest', 'current-indices',
                     'solar-wind/history', 'geomagnetic/history', 'history'),
                    (wind, geo, additional, wind_history, geo_history, hourly)))
    result['sdo-browse'] = {'images': [dict(channel=channel, asset_id=f'2026103012_{channel}_{"a"*64}',
                                          captured_at=NOW.isoformat()) for channel in CHANNELS]}
    return result


def install_sources(monkeypatch, *, missing_forecasts=False):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(summary, 'datetime', Clock)
    data = sources()
    reads = AsyncMock(side_effect=lambda resource, params=None: {'success': True, 'data': data[resource]})
    monkeypatch.setattr(summary, 'read_observations', reads)
    forecasts = {product: release(product) for product in summary.PRODUCTS}
    read = Mock(side_effect=ArtifactNotReadyError('not ready') if missing_forecasts else forecasts.__getitem__)
    monkeypatch.setattr(summary, 'read_release', read)
    monkeypatch.setattr(summary, 'read_event_forecast', Mock(side_effect=ArtifactNotReadyError('not ready')))
    solar = SolarIndices(as_of=NOW, items=[dict(metric=metric, value=100., observed_at=ISSUE-timedelta(days=day))
        for metric in ('f10_7', 's10', 'm10', 'y10') for day in (3, 2, 1, 0)])
    monkeypatch.setattr(summary, 'read_solar_indices', Mock(return_value=solar))
    return data, reads, forecasts


def test_summary_uses_live_observations_and_generates_images_once_per_variable(monkeypatch):
    _, reads, forecasts = install_sources(monkeypatch)
    data, images = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    plasma = data['observations']['plasma']['items']
    assert plasma[0]['value'] == 455.
    assert plasma[0]['observed_at'] == NOW.isoformat()
    assert plasma[0]['forecast_issue_time'] == ISSUE.isoformat()
    assert plasma[0]['risk_level'] == 2
    assert data['observations']['geo']['items'][2]['risk_level'] == 2
    names = [item['image_name'] for group in data['observations'].values() for item in group['items'] if item['image_name']]
    assert len(names) == len(set(names)) == len(images) == 12
    assert all(name in images for name in names)
    assert reads.await_count == 7
    assert data['sdo_images'] == {channel: f'2026103012_{channel}_{"a"*64}' for channel in CHANNELS}
    assert data['cme_events_available'] is False
    svg = ET.fromstring(images[plasma[0]['image_name']])
    circles = svg.findall('{http://www.w3.org/2000/svg}circle')
    assert len(circles) == 7
    assert all(artifact.model_info == {} for forecast in forecasts.values() for artifact in forecast.artifacts)
    assert all(isinstance(item['risk_level'], int) and 0 <= item['risk_level'] <= 4 and 'quality' not in item
               for group in data['observations'].values() for item in group['items'])


def test_quantiles_win_over_thresholds_and_current_point_is_not_old_forecast_input(monkeypatch):
    install_sources(monkeypatch)
    renderer = summary.ForecastImageGenerator.render
    calls = []
    def render(self, before, current, after, levels):
        calls.append((current, after))
        return renderer(self, before, current, after, levels)
    monkeypatch.setattr(summary.ForecastImageGenerator, 'render', render)
    asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    speed = next((current, after) for current, after in calls if current.value == 455.)
    assert speed[0].time == NOW
    assert [point.value for point in speed[1]] == [500., 500., 500.]
    kp = next((current, after) for current, after in calls if current.value == 4.5)
    assert [point.time for point in kp[1]] == [ISSUE+timedelta(hours=i) for i in (3, 6)]
    southward = next((current, after) for current, after in calls if current.value == -12.5)
    assert [point.value for point in southward[1]] == [-7.5]*3


@pytest.mark.parametrize('catalog', [{}, {'images': []}, {'images': [{'channel': '094', 'asset_id': '../invalid'}]}])
def test_missing_or_invalid_sdo_catalog_does_not_hide_observations(monkeypatch, catalog):
    data, _, _ = install_sources(monkeypatch)
    data['sdo-browse'] = catalog
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    assert result['sdo_images'] == {}
    assert result['observations']['plasma']['items'][0]['value'] == 455.


def test_unavailable_forecasts_keep_observations_without_fake_charts(monkeypatch):
    install_sources(monkeypatch, missing_forecasts=True)
    data, images = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    assert images == {}
    assert data['observations']['plasma']['items'][0]['value'] == 455.
    assert all(item['image_name'] is None for group in data['observations'].values() for item in group['items'])


def test_radio_uses_calibrated_observations_daily_charts_and_configured_colors(monkeypatch):
    install_sources(monkeypatch)
    calls = []
    renderer = summary.ForecastImageGenerator.render
    def render(self, before, current, after, levels):
        calls.append((before, current, after, levels))
        return renderer(self, before, current, after, levels)
    monkeypatch.setattr(summary.ForecastImageGenerator, 'render', render)
    data, images = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    radio = data['observations']['radio']['items']
    assert [item['value'] for item in radio] == [150., 100., 100., 100.]
    assert all(item['measurements'] == 'sfu' and item['image_name'] in images for item in radio)
    assert [item['risk_level'] for item in radio] == [2, 2, 2, 2]
    for metric in ('f10_7', 's10', 'm10', 'y10'):
        before, current, after, _ = next(call for call in calls if call[3] == LEVELS[metric])
        assert [p.time for p in before] == [ISSUE-timedelta(days=i) for i in (3, 2, 1)]
        assert current.time == ISSUE
        assert [p.time for p in after] == [ISSUE+timedelta(days=i) for i in (1, 2, 3)]


def test_missing_solar_estimates_does_not_hide_other_observations(monkeypatch):
    install_sources(monkeypatch)
    monkeypatch.setattr(summary, 'read_solar_indices', Mock(side_effect=ArtifactNotReadyError('not ready')))
    data, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    radio = data['observations']['radio']['items']
    assert radio[0]['value'] == 150.
    assert all(item['value'] is None and item['image_name'] is None for item in radio[1:])
    assert data['observations']['plasma']['items'][0]['image_name'] is not None


def test_bad_quality_and_missing_history_do_not_generate_replacement_values(monkeypatch):
    sources, _, _ = install_sources(monkeypatch)
    sources['solar-wind/latest']['series']['v']['latest']['quality'] = 'flagged'
    sources['solar-wind/history']['series']['n']['points'] = []
    data, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    speed, density, _ = data['observations']['plasma']['items']
    assert speed['value'] == 455. and speed['risk_level'] == 2 and speed['image_name'] is None
    assert density['value'] == 5. and density['image_name'] is None


def test_distant_forecast_peaks_do_not_increase_summary_risk(monkeypatch):
    data, _, forecasts = install_sources(monkeypatch)
    forecasts['dst'] = release('dst', future_peak=-200.)
    forecasts['solar-wind-density'] = release('solar-wind-density', future_peak=95.)
    forecasts['solar-wind-speed'] = release('solar-wind-speed', future_peak=750.)
    data['solar-wind/history']['series']['n']['points'] = []
    data['geomagnetic/latest']['series']['dst']['latest'] = None
    data['solar-wind/history']['series']['v']['points'][0]['value'] = 9999.
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    speed, density, _ = result['observations']['plasma']['items']
    dst = result['observations']['geo']['items'][2]
    assert speed['risk_level'] == dst['risk_level'] == 2
    assert density['risk_level'] == 0
    assert density['image_name'] is None and dst['image_name'] is None


@pytest.mark.parametrize('missing', ['none', 'history', 'current'])
@pytest.mark.parametrize('peak_lead,expected', [(1, 4), (2, 4), (3, 4), (4, 0), (12, 0)])
def test_speed_risk_uses_only_current_and_three_hourly_forecast_points(monkeypatch, missing, peak_lead, expected):
    data, _, forecasts = install_sources(monkeypatch)
    data['solar-wind/latest']['series']['v']['latest']['value'] = 325.
    forecasts['solar-wind-speed'] = release('solar-wind-speed', speed=350., future_peak=750., peak_lead=peak_lead)
    if missing == 'history':
        data['solar-wind/history']['series']['v']['points'] = []
    elif missing == 'current':
        data['solar-wind/latest']['series']['v']['latest'] = None
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    speed = result['observations']['plasma']['items'][0]
    assert speed['risk_level'] == expected
    if missing != 'none':
        assert speed['image_name'] is None


def test_old_observation_does_not_force_chart_targets_into_the_past(monkeypatch):
    data, _, _ = install_sources(monkeypatch)
    data['geomagnetic/latest']['series']['kp']['latest'] = {
        'interval_start': (ISSUE-timedelta(hours=3)).isoformat(), 'value': 4.3, 'quality': 'unverified'}
    result, images = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    kp = result['observations']['geo']['items'][0]
    assert kp['image_name'] in images
    assert kp['observed_at'] == (ISSUE-timedelta(hours=3)).isoformat()


@pytest.mark.parametrize('peak_lead,included', [(1, False), (2, False), (3, True), (6, True), (9, False), (12, False)])
def test_kp_ap_risk_uses_only_current_and_two_three_hour_forecast_points(monkeypatch, peak_lead, included):
    _, _, forecasts = install_sources(monkeypatch)
    forecasts['geomagnetic-activity'] = release('geomagnetic-activity', future_peak=150.,
                                               peak_lead=peak_lead, threshold_peak_lead=peak_lead)
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    kp, ap, _ = result['observations']['geo']['items']
    assert kp['risk_level'] == (4 if included else 1)
    assert ap['risk_level'] == (4 if included else 2)


@pytest.mark.parametrize('value,expected', [(449, 0), (450, 1), (500, 2), (600, 3), (700, 4), (-700, 4)])
def test_risk_levels_use_absolute_values_and_all_five_colors(value, expected):
    assert summary.risk_level({'value': value}, [], LEVELS['v']) == expected
    assert summary.risk_level(None, [value], LEVELS['v']) == expected


def test_past_history_does_not_increase_risk(monkeypatch):
    data, _, _ = install_sources(monkeypatch)
    data['solar-wind/history']['series']['v']['points'][0]['value'] = 9999.
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    assert result['observations']['plasma']['items'][0]['risk_level'] == 2


def test_negative_field_component_without_forecast_still_has_a_risk(monkeypatch):
    data, _, _ = install_sources(monkeypatch)
    data['solar-wind/latest']['series']['bx']['latest']['value'] = -25.
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
    assert result['observations']['hmf']['items'][1]['risk_level'] == 4


def test_cme_age_filter_and_exact_public_fields(monkeypatch):
    install_sources(monkeypatch)
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
    monkeypatch.setattr(summary, 'read_event_forecast', Mock(return_value=SimpleNamespace(events=events)))
    result, _ = asyncio.run(summary.build_summary(SummarySettings(critical_levels=LEVELS)))
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
    assert bool(summary.cme_events(SimpleNamespace(events=[event]), NOW)) is expected


def test_cme_grace_period_is_configurable():
    event = SolarWindEvent(event_id='cme', kind='cme', status='prediction_elapsed', source='DONKI/WSA-ENLIL',
        available_at=NOW, observed_at=NOW-timedelta(days=2),
        arrival_at=NOW-timedelta(hours=6), arrival_target='Earth')
    events = SimpleNamespace(events=[event])
    assert summary.cme_events(events, NOW, grace_hours=6)
    assert not summary.cme_events(events, NOW, grace_hours=5)


@pytest.mark.parametrize('probabilities,expected', [
    ([.1, .05, .01], 3.), ([.8, .2, .1], 4.5), ([.9, .8, .1], 5.5), ([.9, .8, .7], 7.),
    ([.2, .8, .1], 3.), ([.5, .5, .5], 3.),
])
def test_modal_threshold_interval(probabilities, expected):
    assert summary.threshold_value(probabilities, (4, 5, 6)) == expected


@pytest.mark.parametrize('configured', [
    {'ttl_seconds': 601}, {'ttl_seconds': 600, 'image_ttl_seconds': 600},
    {'critical_levels': {'kp': []}}, {'critical_levels': {'kp': [4, 6, 5]}},
])
def test_invalid_cache_or_chart_settings(configured):
    with pytest.raises(ValueError):
        SummarySettings(**configured)


def test_public_routes_deliver_cached_json_and_svg_with_conditional_get(monkeypatch):
    monkeypatch.setattr(routes, 'cached_summary', AsyncMock(return_value=b'{"success":true,"data":{}}'))
    image = AsyncMock(return_value=(b'<svg/>', 1200))
    monkeypatch.setattr(routes, 'read_image', image)
    app = FastAPI()
    app.include_router(routes.router)
    async def verify():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://api') as client:
            response = await client.get('/summary')
            assert response.json() == {'success': True, 'data': {}}
            assert response.headers['cache-control'] == 'no-store'
            path = '/summary/images/'+'a'*64+'.svg'
            response = await client.get(path)
            assert response.content == b'<svg/>' and response.headers['content-type'] == 'image/svg+xml'
            assert response.headers['cache-control'] == 'public, max-age=1200, immutable'
            cached = await client.get(path, headers={'If-None-Match': response.headers['etag']})
            assert cached.status_code == 304 and not cached.content
    asyncio.run(verify())


def test_invalid_image_names_do_not_access_redis(monkeypatch):
    monkeypatch.setattr(summary, 'get_cache', Mock(side_effect=AssertionError('Unexpected cache access')))
    with pytest.raises(HTTPException) as error:
        asyncio.run(summary.read_image('../outside.svg'))
    assert error.value.status_code == 404
