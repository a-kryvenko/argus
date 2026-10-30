"""Compose and cache the homepage; SVGs are disposable Redis entries."""
import asyncio
from bisect import bisect_right
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import logging
import math
import os
import re

from fastapi import HTTPException
from redis.asyncio import Redis
from redis.exceptions import RedisError

from common.config import get_config
from common.schemas.forecast_release import PREDICTION_COLUMNS
from common.schemas.response import success_response
from common.schemas.sdo_browse import SDOBrowseCatalog
from api.schemas.summary import ChartPoint, IMAGE_NAME_PATTERN, SummarySettings
from api.services.errors import ArtifactNotReadyError
from api.services.forecasts.client import read_release, read_event_forecast, read_solar_indices
from api.services.observations.client import read_observations
from api.services.summary_images import ForecastImageGenerator

logger = logging.getLogger(__name__)
SUMMARY_KEY = 'api:summary:v9'
IMAGE_KEY = 'api:summary:image:'
_cache = None
GROUPS = {
    'plasma': ('Plasma', (('v', 'v', 'km/s'), ('n', 'n', 'cm⁻³'), ('t', 't', 'K'))),
    'geo': ('Geomagnetic', (('kp', 'Kp', ''), ('ap', 'Ap', ''), ('dst', 'Dst', 'nT'))),
    'hmf': ('HMF', tuple((m, m.capitalize(), 'nT') for m in ('bt', 'bx', 'by', 'bz'))),
    'radio': ('Radio', tuple((m, label, 'sfu') for m, label in (
        ('f10_7', 'F10.7'), ('s10', 'S10'), ('m10', 'M10'), ('y10', 'Y10')))),
}
PRODUCTS = ('solar-wind-speed', 'solar-wind-density', 'solar-wind-temperature',
            'geomagnetic-activity', 'dst', 'hmf', 'solar-radiation')


def settings():
    return SummarySettings.model_validate(get_config().project_config.get('api', {}).get('summary', {}))


def get_cache():
    global _cache
    if _cache is None:
        url = os.getenv('API_REDIS_URL')
        if not url:
            raise HTTPException(503, 'Summary cache is not configured')
        _cache = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=3)
    return _cache


async def close_cache():
    global _cache
    if _cache is not None:
        await _cache.aclose()
        _cache = None


def timestamp(value):
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.tzinfo is None:
        raise ValueError('Observation and forecast timestamps must include a timezone')
    return stamp.astimezone(UTC)


def sample(point, now):
    if not point or point.get('value') is None:
        return None
    try:
        value = float(point['value'])
        time = timestamp(point.get('observed_at', point.get('interval_start', point.get('issue_time'))))
    except (ValueError, TypeError, AttributeError):
        return None
    if not math.isfinite(value) or time > now:
        return None
    return {'time': time, 'value': value, 'quality': point.get('quality')}


async def observation_read(resource, params=None, *, optional=False):
    try:
        response = await read_observations(resource, params)
        if not isinstance(response, dict) or response.get('success') is not True or not isinstance(response.get('data'), dict):
            raise HTTPException(503, 'Observation service returned an invalid response')
        if resource in ('solar-wind/latest', 'geomagnetic/latest', 'current-indices') and not isinstance(response['data'].get('series'), dict):
            raise HTTPException(503, 'Observation service returned an invalid response')
        return response['data']
    except HTTPException:
        if not optional:
            raise
        logger.warning('Summary observations are unavailable: %s', resource)
        return {}


async def sdo_images_read():
    data = await observation_read('sdo-browse', optional=True)
    try:
        return {image.channel: image.asset_id for image in SDOBrowseCatalog.model_validate(data).images}
    except ValueError:
        logger.warning('Summary SDO images are unavailable')
        return {}


async def forecast_read(product):
    try:
        if product == 'events':
            return await asyncio.to_thread(read_event_forecast)
        return await asyncio.to_thread(read_release, product)
    except ArtifactNotReadyError:
        logger.warning('Summary forecast is unavailable: %s', product)
        return None


async def solar_indices_read():
    try:
        return await asyncio.to_thread(read_solar_indices)
    except ArtifactNotReadyError:
        logger.warning('Summary calibrated solar indices are unavailable')
        return None


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


def risk_level(point, future, levels):
    values = ([point['value']] if point else []) + list(future)
    magnitudes = [abs(value) for value in values if math.isfinite(value)]
    if not magnitudes:
        return 0
    peak = max(magnitudes)
    return min(4, sum(peak >= abs(level) for level in levels))


def render_charts(releases, current, histories, levels, now):
    selected = {}
    for release in releases:
        if release is None:
            continue
        for artifact in release.artifacts:
            variable = chart_variable(artifact)
            if variable and (variable[0] not in selected or variable[1]):
                selected[variable[0]] = (release, artifact, variable)
    references, images, outlooks = {}, {}, {}
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
                before.append(ChartPoint(time=time, value=display(history[time])))
        if len(before) < 2 or len(after) < 2:
            logger.info('Summary chart %s has insufficient history or future points', metric)
            continue
        content = ForecastImageGenerator().render(before,
            ChartPoint(time=latest['time'], value=display(latest['value'])),
            [ChartPoint(time=time, value=value) for time, value in sorted(after.items())], levels[metric])
        identity = f'{release.release_id}:{metric}:{latest["time"].isoformat()}'.encode()
        name = hashlib.sha256(identity+b'\0'+content).hexdigest()+'.svg'
        images[name] = content
        references[metric] = {'image_name': name, 'forecast_issue_time': release.issue_time.isoformat()}
    return references, images, outlooks


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


async def build_summary(config):
    now = datetime.now(UTC)
    bounds = {'from': now-timedelta(hours=12), 'to': now}
    wind, geo, additional, wind_history, geo_history, hourly, sdo_images, solar, *forecasts = await asyncio.gather(
        observation_read('solar-wind/latest'), observation_read('geomagnetic/latest'),
        observation_read('current-indices'),
        observation_read('solar-wind/history', {**bounds, 'resolution': '1h'}, optional=True),
        observation_read('geomagnetic/history', bounds, optional=True),
        observation_read('history', {**bounds, 'limit': 13}, optional=True),
        sdo_images_read(),
        solar_indices_read(),
        *(forecast_read(product) for product in (*PRODUCTS, 'events')))
    now = datetime.now(UTC)
    current, histories = {}, {}
    for data in (wind, geo):
        for metric, series in data['series'].items():
            current[metric] = sample(series['latest'], now)
    current.update({metric: sample(point, now) for metric, point in additional['series'].items()})
    for data in (wind_history, geo_history):
        for metric, series in data.get('series', {}).items():
            histories[metric] = [value for point in series['points'] if (value := sample(point, now)) is not None]
    for point in hourly.get('points', []):
        value = sample({'issue_time': point['issue_time'], 'value': point.get('ap')}, now)
        if value is not None:
            histories.setdefault('ap', []).append(value)
    if solar is not None and solar.as_of <= now:
        for measurement in solar.items:
            metric = measurement.metric
            value = sample(measurement.model_dump(), now)
            if value is not None:
                histories.setdefault(metric, []).append(value)
                if metric != 'f10_7' and (not current.get(metric) or value['time'] > current[metric]['time']):
                    current[metric] = value
    references, images, outlooks = render_charts(forecasts[:-1], current, histories, config.critical_levels, now)
    groups = {}
    for group, (title, metrics) in GROUPS.items():
        items = []
        for metric, label, unit in metrics:
            point = current.get(metric)
            risk = risk_level(point, outlooks.get(metric, []), config.critical_levels.get(metric, []))
            items.append({'title': label, 'risk_level': risk, 'value': point['value'] if point else None,
                'measurements': unit, 'observed_at': point['time'].isoformat() if point else None,
                'image_name': None,
                **references.get(metric, {})})
        groups[group] = {'title': title, 'items': items}
    events = forecasts[-1]
    return {'generated_at': now.isoformat(), 'observations': groups, 'sdo_images': sdo_images,
            'cme_events': cme_events(events, now, config.cme_grace_hours),
            'cme_events_available': events is not None}, images


async def cached_summary():
    cache = get_cache()
    try:
        if (cached := await cache.get(SUMMARY_KEY)) is not None:
            return cached
        async with cache.lock(SUMMARY_KEY+':lock', timeout=240, blocking_timeout=10):
            if (cached := await cache.get(SUMMARY_KEY)) is not None:
                return cached
            config = settings()
            data, images = await build_summary(config)
            payload = json.dumps(success_response(data).model_dump(mode='json'), allow_nan=False, separators=(',', ':')).encode()
            async with cache.pipeline(transaction=True) as pipeline:
                for name, content in images.items():
                    pipeline.set(IMAGE_KEY+name, content, ex=config.image_ttl_seconds)
                pipeline.set(SUMMARY_KEY, payload, ex=config.ttl_seconds)
                await pipeline.execute()
            return payload
    except RedisError:
        raise HTTPException(503, 'Summary cache is unavailable') from None


async def read_image(name):
    if not re.fullmatch(IMAGE_NAME_PATTERN, name):
        raise HTTPException(404, 'Summary image not found')
    try:
        async with get_cache().pipeline(transaction=True) as pipeline:
            pipeline.get(IMAGE_KEY+name).ttl(IMAGE_KEY+name)
            content, ttl = await pipeline.execute()
    except RedisError:
        raise HTTPException(503, 'Summary cache is unavailable') from None
    if content is None:
        raise HTTPException(404, 'Summary image not found')
    return content, max(ttl, 0)
