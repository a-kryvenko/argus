"""Build the public summary, calculate risks and publish SVGs in S3."""
import asyncio
from bisect import bisect_right
import csv
from datetime import UTC, datetime, timedelta
import json
import io
import logging
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from common.config import get_config
from common.data import datamanager
from common.schemas.forecast_release import PREDICTION_COLUMNS
from common.schemas.sdo_browse import SDOBrowseCatalog
from common.schemas.solar_wind_events import SolarWindEventForecast
from prophet.config import load_config
from prophet.services.releases.publication import read_release, ReleaseNotFound
from prophet.services.solar_indices import read_solar_indices

logger = logging.getLogger(__name__)
BUCKET = 'argus-public'
PREFIX = 'forecasts/summary'
PRODUCTS = ('solar-wind-speed', 'solar-wind-density', 'solar-wind-temperature',
            'geomagnetic-activity', 'dst', 'hmf', 'solar-radiation')


GROUPS = {
    'plasma': ('Plasma', (('v', 'v', 'km/s'), ('n', 'n', 'cm⁻³'), ('t', 't', 'K'))),
    'geo': ('Geomagnetic', (('kp', 'Kp', ''), ('ap', 'Ap', ''), ('dst', 'Dst', 'nT'))),
    'hmf': ('HMF', tuple((m, m.capitalize(), 'nT') for m in ('bt', 'bx', 'by', 'bz'))),
    'radio': ('Radio', tuple((m, label, 'sfu') for m, label in (
        ('f10_7', 'F10.7'), ('s10', 'S10'), ('m10', 'M10'), ('y10', 'Y10')))),
}


def render_image(before, current, after, critical_levels) -> bytes:
    width, height, padding = 480, 240, 12
    colors = ('#22c55e', '#eab308', '#f97316', '#ef4444', '#dc143c')
    points = [*before, current, *after]
    values = [point['value'] for point in points]
    times = [point['time'] for point in points]
    if not (2 <= len(before) <= 3 and 2 <= len(after) <= 3):
        raise ValueError('Expected two or three points before and after the current observation')
    if any(time.utcoffset() is None for time in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Point times must be aware and strictly increasing')
    if not all(math.isfinite(value) for value in [*values, *critical_levels]):
        raise ValueError('Values and critical levels must be finite')
    if not 1 <= len(critical_levels) <= 4 or not (
        all(a < b for a, b in zip(critical_levels, critical_levels[1:])) or
        all(a > b for a, b in zip(critical_levels, critical_levels[1:]))
    ):
        raise ValueError('Expected one to four ordered critical levels')
    low, high = min(values), max(values)
    margin = (high-low) * .1 if high > low else max(abs(low) * .05, 1.)
    low, high = low-margin, high+margin

    def y(value):
        return height-padding - (value-low)/(high-low)*(height-2*padding)

    coordinates = [(
        padding + index/(len(points)-1)*(width-2*padding),
        y(point['value']),
    ) for index, point in enumerate(points)]
    levels = critical_levels
    direction = 1 if (levels[-1] > levels[0] if len(levels) > 1 else levels[0] >= 0) else -1
    levels = [value*direction for value in levels]
    # Extend the first danger band down to green; a lone level uses zero as its safe baseline.
    safe = levels[0]-(levels[1]-levels[0] if len(levels) > 1 else max(abs(levels[0]), 1.))
    anchors = [(safe, colors[0]),
               *((level, colors[i+1]) for i, level in enumerate(levels))]
    stops = ''.join(f'<stop offset="{(value-safe)/(levels[-1]-safe):.6f}" stop-color="{color}"/>'
                    for value, color in anchors)
    line = ' '.join(f'{x:.3f},{v:.3f}' for x, v in coordinates)
    current_x = coordinates[len(before)][0]
    dots = ''.join(f'<circle cx="{x:.3f}" cy="{v:.3f}" r="{5 if i == len(before) else 3.5}" '
                   'fill="url(#risk)"/>' for i, (x, v) in enumerate(coordinates))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="240" viewBox="0 0 480 240">'
            f'<defs><linearGradient id="risk" gradientUnits="userSpaceOnUse" x1="0" x2="0" '
            f'y1="{y(safe*direction):.3f}" y2="{y(levels[-1]*direction):.3f}">{stops}'
            '</linearGradient></defs>'
            f'<line x1="{current_x:.3f}" x2="{current_x:.3f}" '
            f'y1="{padding}" y2="{height-padding}" '
            'stroke="white" stroke-width="1.5" vector-effect="non-scaling-stroke" '
            'stroke-dasharray="4 4" stroke-opacity="0.7"/>'
            f'<polyline points="{line}" fill="none" stroke="url(#risk)" stroke-width="2" '
            'vector-effect="non-scaling-stroke" '
            f'stroke-linecap="round" stroke-linejoin="round"/>{dots}</svg>').encode()


def timestamp(value):
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.tzinfo is None:
        raise ValueError('Observation and forecast timestamps must include a timezone')
    return stamp.astimezone(UTC)


def sample(time, value, now):
    time = timestamp(time)
    if value is None or not math.isfinite(float(value)) or time > now:
        return None
    return dict(time=time, value=float(value), quality=None)


def interval_value(index, thresholds):
    if index == 0:
        return thresholds[0]-(thresholds[1]-thresholds[0] if len(thresholds) > 1 else 1.)
    if index == len(thresholds):
        return thresholds[-1]+(thresholds[-1]-thresholds[-2] if len(thresholds) > 1 else 1.)
    return (thresholds[index-1]+thresholds[index])/2


def threshold_value(probabilities, thresholds):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid threshold probabilities')
    survival = [1.]
    for probability in probabilities:
        survival.append(min(survival[-1], probability))
    survival.append(0.)
    masses = [a-b for a, b in zip(survival, survival[1:])]
    return interval_value(max(range(len(masses)), key=masses.__getitem__), thresholds)


def chart_variable(artifact):
    columns = PREDICTION_COLUMNS.get(artifact.name, ())
    median = next((name for name in columns if name.endswith('_q50')), None)
    prefix = median[:-4] if median else columns[0][2:].rsplit('_ge_', 1)[0] if columns and columns[0].startswith('p_') else None
    if prefix is None:
        return None
    metric = {'f107': 'f10_7', 'bs': 'bz'}.get(prefix, prefix)
    thresholds = {float(name.rsplit('_ge_', 1)[1]): name for name in columns if name.startswith('p_')}
    return metric, median, thresholds, prefix == 'bs'


def render_charts(releases, current, histories, levels, now):
    selected = {}
    for release in releases:
        if release is None:
            continue
        for artifact in release.artifacts:
            variable = chart_variable(artifact)
            if variable and (variable[0] not in selected or variable[1]):
                selected[variable[0]] = (release, artifact, variable)
    images, outlooks = {}, {}
    for metric, (release, artifact, (_, median, threshold_columns, southward)) in selected.items():
        thresholds = sorted(threshold_columns)
        future = {}
        for row in csv.DictReader(io.StringIO(artifact.csv_text)):
            valid = timestamp(row['valid_time'])
            if valid <= now:
                continue
            value = float(row[median]) if median else threshold_value(
                [float(row[threshold_columns[t]]) for t in thresholds], thresholds)
            future[valid] = -value if southward else value
        latest = current.get(metric)
        daily = metric in ('f10_7', 's10', 'm10', 'y10')
        cadence, count = (24, 3) if daily else (3, 2) if metric in ('kp', 'ap') else (1, 3)
        anchor = (release.issue_time if daily else latest['time'] if latest else now).replace(
            minute=0, second=0, microsecond=0)
        first = max(1, int((now-anchor).total_seconds() // (cadence*3600))+1)
        targets = {anchor+timedelta(hours=cadence*step) for step in range(first, first+count)}
        after = {time: value for time, value in future.items() if time in targets}
        outlooks[metric] = list(after.values())
        if not latest or latest['quality'] in ('missing', 'flagged') or metric not in levels:
            continue
        def display(value):
            if not thresholds:
                return value
            value = max(-value, 0.) if southward else value
            representative = interval_value(bisect_right(thresholds, value), thresholds)
            return -representative if southward else representative
        history = {point['time']: point['value'] for point in histories.get(metric, [])
                   if point['quality'] not in ('missing', 'flagged') and point['time'] < latest['time']}
        before = []
        for step in range(count, 0, -1):
            target = latest['time']-timedelta(hours=cadence*step)
            times = [time for time in history if target-timedelta(hours=24 if daily else 1) < time <= target]
            if times:
                time = max(times)
                before.append(dict(time=time, value=display(history[time])))
        if len(before) < 2 or len(after) < 2:
            logger.info('Summary chart %s has insufficient history or future points', metric)
            continue
        content = render_image(before,
            dict(time=latest['time'], value=display(latest['value'])),
            [dict(time=time, value=value) for time, value in sorted(after.items())], levels[metric])
        images[metric] = content
    return images, outlooks


def read_observations(now):
    """Use Clio's numeric interface without importing public API presentation code."""
    url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise RuntimeError('Observation service is not configured')
    metrics = 'v,n,t,bx,by,bz,bt,kp,ap,dst,f10_7,s10,m10,y10'
    with httpx.Client(timeout=httpx.Timeout(30, connect=5), follow_redirects=False, trust_env=False) as client:
        def read(resource, **params):
            response = client.get(url.rstrip('/')+'/internal/v1/observations/'+resource,
                                  params=params, headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
            payload = response.json()
            if payload.get('success') is not True or not isinstance(payload.get('data', {}).get('points'), list):
                raise ValueError('Invalid chart observation response')
            return payload['data']['points']
        latest = read('latest', metrics=metrics)
        bounds = {'from': (now-timedelta(hours=12)).isoformat(), 'to': now.isoformat()}
        try:
            hourly = read('history', metrics=metrics, interval='1h', **bounds)
            geo = read('history', metrics='kp,dst', interval='1m', **bounds)
        except (httpx.HTTPError, ValueError):
            logger.warning('Summary observation history is unavailable')
            hourly, geo = [], []
    current, histories = {}, {}
    for row in latest:
        for metric in metrics.split(','):
            if metric in row and (point := sample(row['time'], row[metric], now)) is not None:
                if metric not in current or point['time'] > current[metric]['time']:
                    current[metric] = point
    for rows, selected in ((hourly, set(metrics.split(','))-{'kp', 'dst'}), (geo, {'kp', 'dst'})):
        for row in rows:
            for metric in selected:
                if metric in row and (point := sample(row['time'], row[metric], now)) is not None:
                    histories.setdefault(metric, []).append(point)
    return current, histories




def risk_level(point, future, levels):
    values = ([point['value']] if point else []) + list(future)
    magnitudes = [abs(value) for value in values if math.isfinite(value)]
    if not magnitudes:
        return 0
    peak = max(magnitudes)
    return min(4, sum(peak >= abs(level) for level in levels))


def cme_events(events, now, grace_hours=24):
    fields = {'event_id', 'source_url', 'observed_at',
              'arrival_window_start', 'arrival_window_end'}
    cutoff = now-timedelta(days=7)
    return [event.model_dump(mode='json', include=fields) for event in events.events
            if event.kind == 'cme' and cutoff <= (event.observed_at or event.available_at) <= now
            and event.available_at <= now
            and event.status in ('predicted_hit', 'prediction_elapsed')
            and event.arrival_target == 'Earth' and event.arrival_at is not None
            and (event.arrival_window_end or event.arrival_at)+timedelta(hours=grace_hours) >= now] if events else []


def read_sdo_images():
    url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    try:
        if not url or not token:
            raise ValueError('Observation service is not configured')
        with httpx.Client(timeout=httpx.Timeout(30, connect=5), follow_redirects=False, trust_env=False) as client:
            response = client.get(url.rstrip('/')+'/internal/v1/observations/sdo',
                                  headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
        payload = response.json()
        if payload.get('success') is not True:
            raise ValueError('Invalid SDO response')
        return {image.channel: dict(asset_id=image.asset_id, urls=image.urls)
                for image in SDOBrowseCatalog.model_validate(payload['data']).images}
    except (httpx.HTTPError, ValueError, KeyError):
        logger.warning('Summary SDO images are unavailable')
        return {}


async def read_event_forecast():
    try:
        release = await read_release('solar-wind-events')
    except ReleaseNotFound:
        return None
    row = next(csv.DictReader(io.StringIO(release.artifacts[0].csv_text)))
    return SolarWindEventForecast.model_validate_json(row['report_json'])


async def build_summary():
    config = load_config().summary
    now = datetime.now(UTC)
    (current, histories), solar, sdo_images, events = await asyncio.gather(
        asyncio.to_thread(read_observations, now), read_solar_indices(),
        asyncio.to_thread(read_sdo_images), read_event_forecast())
    if solar is not None and solar.as_of <= now:
        for measurement in solar.items:
            metric = measurement.metric
            point = sample(measurement.observed_at, measurement.value, now)
            if point is not None:
                histories.setdefault(metric, []).append(point)
                if metric != 'f10_7' and (metric not in current or point['time'] > current[metric]['time']):
                    current[metric] = point
    releases = []
    for product in PRODUCTS:
        try:
            releases.append(await read_release(product))
        except ReleaseNotFound:
            logger.warning('Summary forecast is unavailable: %s', product)
    images, outlooks = render_charts(releases, current, histories, config.critical_levels, now)
    groups = {}
    for group, (title, metrics) in GROUPS.items():
        items = []
        for metric, label, unit in metrics:
            point = current.get(metric)
            items.append({'title': label,
                'risk_level': risk_level(point, outlooks.get(metric, []), config.critical_levels.get(metric, [])),
                'value': point['value'] if point else None, 'measurements': unit,
                'observed_at': point['time'].isoformat() if point else None, 'image_url': None})
        groups[group] = {'title': title, 'items': items, 'risk_level': max(item['risk_level'] for item in items)}
    return {'generated_at': now.isoformat(), 'observations': groups, 'sdo_images': sdo_images,
            'cme_events': cme_events(events, now, config.cme_grace_hours),
            'cme_events_available': events is not None}, images


async def publish_summary():
    data, images = await build_summary()
    base = os.environ['S3_PUBLIC_BASE_URL'].rstrip('/')
    version = datetime.now(UTC).strftime('%Y%m%d%H%M')
    links = {}
    for metric, content in images.items():
        name = f'{metric}-{version}.svg'
        key = await asyncio.to_thread(datamanager.upload_file, content, BUCKET, PREFIX+'/images', name,
                                      content_type='image/svg+xml', cache_control='public, max-age=86400, immutable')
        links[metric] = base+'/'+key
    for group, (_, metrics) in GROUPS.items():
        for item, (metric, _, _) in zip(data['observations'][group]['items'], metrics):
            item['image_url'] = links.get(metric)
    path = get_config().data_root/'prophet/summary/latest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, allow_nan=False).encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    logger.info('Published summary with %s charts', len(images))


def read_summary():
    return json.loads((get_config().data_root/'prophet/summary/latest.json').read_bytes())
