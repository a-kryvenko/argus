"""API-owned selection, presentation and trends over Clio's compact numeric reads."""
import asyncio
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from common.config import get_config
from api.services.observations.client import read_observations as read
from api.services.observations.coverage import coverage
from api.services.observations.trends import change_one_hour, interval_change, value_change

WIND = ('bx', 'by', 'bz', 'bt', 'v', 'n', 't')
INDICES = ('kp', 'dst')
HOURLY = ('ap', 'f10_7', 's10', 'm10', 'y10')
DURATIONS = {'kp': 10800, 'ap': 10800, 'dst': 3600}
LABELS = {'bx': 'Bx', 'by': 'By', 'bz': 'Bz', 'bt': 'Total magnetic field',
          'v': 'Solar wind speed', 'n': 'Proton density', 't': 'Proton temperature',
          'kp': 'Estimated Kp', 'dst': 'Real-time Dst'}
UNITS = {'v': 'km/s', 'n': 'cm⁻³', 't': 'K', 'kp': ''}
SOURCES = {
    'mag': 'https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json',
    'plasma': 'https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json',
    'kp': 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
    'dst': 'https://services.swpc.noaa.gov/products/kyoto-dst.json',
}


def aware(value):
    time = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if time.tzinfo is None:
        raise ValueError('An aware observation timestamp is required')
    return time.astimezone(UTC)


def freshness(metric):
    policies = get_config().project_config['clio']['observations']
    if metric == 'bt':
        return min(policies[m]['freshness_minutes'] for m in ('bx', 'by', 'bz')) * 60
    return policies[metric]['freshness_minutes'] * 60


def metadata(metric, seconds=60):
    index = metric in INDICES
    kind = metric if index else 'mag' if metric in ('bx', 'by', 'bz', 'bt') else 'plasma'
    result = {'label': LABELS[metric], 'unit': UNITS.get(metric, 'nT'),
        'source': 'WDC Kyoto via NOAA SWPC' if metric == 'dst' else 'NOAA SWPC' if index else 'NOAA SWPC RTSW',
        'source_url': SOURCES[kind], 'time_basis': 'source_interval_start' if index else 'measurement',
        'resolution_seconds': DURATIONS[metric] if index else seconds,
        'stale_after_seconds': freshness(metric)}
    if index:
        result.update(data_status='estimated' if metric == 'kp' else 'realtime', freshness_basis='interval_end',
                      gap_filling='none', poll_seconds=300 if metric == 'dst' else 60)
    else:
        result.update(location='L1', propagated=False, selection='NOAA active spacecraft',
                      aggregation='normalized_hourly' if seconds == 3600 else 'mean' if seconds == 300 else 'source_1_minute',
                      coordinate_system='GSM' if metric in ('bx', 'by', 'bz') else None)
    return result


async def numeric(resource, params=None):
    payload = await read(resource, params)
    if isinstance(payload, JSONResponse) or not payload.get('success') or not isinstance(payload.get('data'), dict):
        raise HTTPException(503, 'Observation service is unavailable')
    data = payload['data']
    if not isinstance(data.get('points'), list):
        raise HTTPException(503, 'Observation service returned an invalid response')
    return data['points']


def samples(rows, metrics, *, intervals=False):
    result = {metric: [] for metric in metrics}
    for row in rows:
        time = aware(row['time'])
        for metric in metrics:
            if metric not in row:
                continue
            value = row[metric]
            point = {'observed_at': time, 'value': value, 'quality': 'missing' if value is None else None}
            if intervals and metric in DURATIONS:
                point.update(interval_start=time, interval_end=time+timedelta(seconds=DURATIONS[metric]))
                point['interval_status'] = 'completed' if point['interval_end'] <= datetime.now(UTC) else 'in_progress'
            result[metric].append(point)
    return result


def latest_series(rows, metrics, now):
    grouped = samples(rows, metrics, intervals=True)
    series = {}
    for metric in metrics:
        point = grouped[metric][-1] if grouped[metric] else None
        if metric in WIND or metric in INDICES:
            index = metric in INDICES
            age = max(0, int((now - point['interval_end' if index else 'observed_at']).total_seconds())) if point else None
            series[metric] = {**metadata(metric), 'latest': point,
                'lag_seconds' if index else 'age_seconds': age,
                'status': 'missing' if point is None or point['value'] is None else 'stale' if age > freshness(metric) else 'fresh'}
        else:
            series[metric] = point
    return series


async def latest(metrics, now):
    rows = await numeric('latest', {'metrics': ','.join(metrics)})
    return {'generated_at': now, 'series': latest_series(rows, metrics, now)}


async def history(metrics, start, end, interval, now, *, indices=False):
    rows = await numeric('metrics', {'metrics': ','.join(metrics), 'from': start-timedelta(hours=3) if indices else start,
                                     'to': end, 'interval': '1m' if indices else interval})
    seconds = {'1m': 60, '5m': 300, '1h': 3600}[interval]
    grouped = samples(rows, metrics, intervals=indices)
    series = {}
    for metric in metrics:
        points = grouped[metric]
        resolution = DURATIONS[metric] if indices else seconds
        if indices:
            points = [p for p in points if p['interval_start'] < end and p['interval_end'] > start]
        item = {**metadata(metric, seconds), 'points': points,
                'coverage': coverage(points, start, end, resolution, intervals=indices, now=now)}
        if not indices and interval != '1m':
            item['coverage']['basis'] = 'available_aggregate_windows'
        series[metric] = item
    return {'from': start, 'to': end, 'series': series, 'resolution_seconds': seconds,
            'gap_filling': 'normalized' if interval == '1h' and not indices else 'none'}


async def read_observations(resource, params=None):
    params = params or {}
    now = datetime.now(UTC)
    if resource == 'sdo-browse':
        return await read(resource, params)
    if resource in ('solar-wind/latest', 'geomagnetic/latest', 'current-indices'):
        defaults = WIND if resource.startswith('solar-wind') else INDICES if resource.startswith('geomagnetic') else HOURLY
        metrics = params.get('metrics', ','.join(defaults)).split(',')
        data = await latest(metrics, now)
        if resource == 'current-indices' and params.get('trends'):
            old = await numeric('latest', {'metrics': ','.join(metrics), 'as_of': now-timedelta(hours=1)})
            previous = samples(old, metrics)
            data['changes_1h'] = {m: value_change(data['series'][m], previous[m][-1] if previous[m] else None) for m in metrics}
    elif resource in ('solar-wind/history', 'geomagnetic/history'):
        end = aware(params['to']) if params.get('to') else now
        start = aware(params['from']) if params.get('from') else end-timedelta(hours=24)
        indices = resource.startswith('geomagnetic')
        metrics = params.get('metrics', ','.join(INDICES if indices else WIND)).split(',')
        data = await history(metrics, start, end, params.get('resolution', '1m'), now, indices=indices)
    elif resource == 'history':
        end = aware(params['to']) if params.get('to') else now
        start = aware(params['from']) if params.get('from') else end-timedelta(hours=params.get('limit', 24))
        metrics = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', *HOURLY)
        rows = await numeric('metrics', {'metrics': ','.join(metrics), 'from': start, 'to': end, 'interval': '1h'})
        data = {'points': [{'issue_time': row['time'], **{m: row.get(m) for m in metrics}} for row in rows][-params.get('limit', 24):]}
    elif resource == 'summary':
        metrics = (*WIND, *INDICES)
        current, past = await asyncio.gather(latest(metrics, now), history(metrics, now-timedelta(hours=7), now, '1m', now))
        changes = {}
        for metric in WIND:
            changes[metric] = {**change_one_hour(past['series'][metric]['points'], now, freshness_minutes=freshness(metric)/60),
                               'unit': UNITS.get(metric, 'nT')}
        for metric in INDICES:
            points = samples([{'time': p['observed_at'], metric: p['value']} for p in past['series'][metric]['points']], [metric], intervals=True)[metric]
            changes[metric] = {**(interval_change(points, now) if current['series'][metric]['status'] == 'fresh'
                                  else {'status': 'unavailable', 'value': None, 'reason': 'stale'}), 'unit': UNITS.get(metric, 'nT')}
        data = {'generated_at': now, 'solar_wind': {m: current['series'][m] for m in WIND},
                'geomagnetic': {m: current['series'][m] for m in INDICES}, 'changes_1h': changes, 'lookback_minutes': 75}
    else:
        raise ValueError('Unsupported observation view')
    return {'success': True, 'data': data, 'error': None}
