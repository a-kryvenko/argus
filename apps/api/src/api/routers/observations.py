import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from api.schemas.metadata import MetaQuery
from api.schemas.observations import (
    CurrentObservations,
    CurrentAdditionalIndices,
    IndexHistory,
    IndexLatest,
    ObservationHistory,
    ObservationSummary,
    SolarHistory,
    SolarLatest,
)
from api.services.observations.client import read_observations
from api.services.observations.responses import public_observations
from common.schemas.response import ApiResponse, success_response

router = APIRouter(prefix='/observations', tags=['observations'])
WIND = ('bx', 'by', 'bz', 'bt', 'v', 'n', 't')
INDICES = ('kp', 'dst')
HOURLY = ('ap', 'f10_7', 's10', 'm10', 'y10')
METRICS = (*WIND, *INDICES, *HOURLY)
CURRENT_METRICS = METRICS


def parse_observations(observations: str | None, supported: tuple[str, ...]) -> list[str]:
    selected = list(supported) if observations is None else list(dict.fromkeys(
        value.strip() for value in observations.split(',')))
    if not selected or any(value not in supported for value in selected):
        raise HTTPException(422, 'Unknown observation. Supported: ' + ','.join(supported))
    return selected


def selected_current_observations(observations: str | None = Query(
    default=None, description='Comma-separated latest native observations: ' + ','.join(CURRENT_METRICS),
)) -> list[str]:
    return parse_observations(observations, CURRENT_METRICS)


def selected_observations(observations: str | None = Query(
    default=None, description='Comma-separated observations: ' + ','.join(METRICS),
)) -> list[str]:
    return parse_observations(observations, METRICS)


def payload(result):
    if isinstance(result, JSONResponse):
        return None, result
    result = (result.model_dump(mode='json', by_alias=True) if isinstance(result, ApiResponse)
              else jsonable_encoder(result, by_alias=True))
    if not result.get('success') or result.get('data') is None:
        return None, result
    return result['data'], None


@router.get('', response_model=ApiResponse[CurrentObservations], response_model_exclude_unset=True)
async def current_observations(response: Response, meta: MetaQuery = False,
                               trends: bool = Query(default=False, description='Include observed one-hour changes'),
                               selected: list[str] = Depends(selected_current_observations)):
    response.headers['Cache-Control'] = 'no-store'
    data = {}
    trend_data = None
    if trends and set(selected) & set((*WIND, *INDICES)):
        result = public_observations(await read_observations('summary'), ObservationSummary, meta=meta)
        trend_data, error = payload(result)
        if error is not None:
            return error
        data['changes_1h'] = {key: value['value'] for key, value in trend_data['changes_1h'].items() if key in selected}
    for group, metrics, resource, model in (
        ('solar_wind', WIND, 'solar-wind/latest', SolarLatest),
        ('geomagnetic', INDICES, 'geomagnetic/latest', IndexLatest),
    ):
        requested = [key for key in selected if key in metrics]
        if not requested:
            continue
        if trend_data is not None:
            current = {'generated_at': trend_data['generated_at'], 'series': trend_data[group]}
            if meta:
                current['meta'] = {'series': trend_data['meta'][group]}
        else:
            params = {'metrics': ','.join(requested)} if group == 'solar_wind' else None
            result = public_observations(await read_observations(resource, params), model, meta=meta)
            current, error = payload(result)
            if error is not None:
                return error
        data.update({key: value['latest'] for key, value in current['series'].items() if key in requested})
        if meta:
            data.setdefault('meta', {'series': {}})['series'].update({
                key: value for key, value in current['meta']['series'].items() if key in requested})
    additional = [key for key in selected if key in HOURLY]
    if additional:
        params = {'metrics': ','.join(additional)}
        if trends:
            params['trends'] = True
        result = public_observations(await read_observations('current-indices', params),
                                     CurrentAdditionalIndices, meta=False)
        current, error = payload(result)
        if error is not None:
            return error
        data.update({key: value for key, value in current['series'].items() if key in additional})
        if trends:
            changes = current.get('changes_1h', {})
            data.setdefault('changes_1h', {}).update({key: changes.get(key) for key in additional})
    return success_response(data)


@router.get('/history', response_model=ApiResponse[ObservationHistory])
async def observation_history(
    response: Response,
    meta: MetaQuery = False,
    selected: list[str] = Depends(selected_observations),
    limit: int = Query(default=24, ge=1, le=168),
    start: datetime | None = Query(default=None, alias='from'),
    end: datetime | None = Query(default=None, alias='to'),
    resolution: Literal['1m', '5m', '1h', 'auto'] = Query(
        default='1m', description='auto: 1m through 6 hours, 5m through 24 hours, otherwise 1h',
    ),
):
    """Selected native series and hourly indices. Missing measurements remain gaps."""
    response.headers['Cache-Control'] = 'no-store'
    bounded = start is not None or end is not None
    end = end or datetime.now(UTC)
    start = start or end - timedelta(hours=24)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise HTTPException(422, 'from and to must include a timezone and define a positive interval')
    wind = [key for key in selected if key in WIND]
    indices = [key for key in selected if key in INDICES]
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    interval = end - start
    reads = []
    if wind:
        if resolution == 'auto':
            resolution = '1m' if interval <= timedelta(hours=6) else '5m' if interval <= timedelta(hours=24) else '1h'
        days = {'1m': 7, '5m': 31, '1h': 366}[resolution]
        if interval > timedelta(days=days):
            raise HTTPException(422, f'History interval must be at most {days} days for {resolution}')
        reads.append(('solar-wind/history', SolarHistory, {
            'from': start, 'to': end, 'resolution': resolution, 'metrics': ','.join(wind),
        }))
    if indices:
        if interval > timedelta(days=31):
            raise HTTPException(422, 'Geomagnetic history interval must be at most 31 days')
        reads.append(('geomagnetic/history', IndexHistory, {'from': start, 'to': end}))
    results = await asyncio.gather(*(read_observations(resource, params) for resource, _, params in reads))
    data = {'from': start, 'to': end, 'series': {}}
    for (_, model, _), result in zip(reads, results):
        result = public_observations(result, model, meta=meta)
        if not isinstance(result, ApiResponse) or not result.success or result.data is None:
            return result
        part = result.data
        data['series'].update({key: value for key, value in part.series.items() if key in selected})
        if part.meta is not None:
            metadata = data.setdefault('meta', {'series': {}})
            metadata['series'].update({key: value for key, value in part.meta.series.items() if key in selected})
            metadata.update(part.meta.model_dump(exclude={'series'}, exclude_none=True))
        data.update(part.model_dump(exclude={'series', 'meta'}, exclude_none=True, by_alias=True))
    if set(selected) & set(HOURLY):
        params = {'limit': limit}
        if bounded:
            params.update({'from': start, 'to': end})
        hourly, error = payload(await read_observations('history', params))
        if error is not None:
            return error
        data['points'] = [{key: value for key, value in point.items() if key == 'issue_time' or key in selected}
                          for point in hourly['points']
                          if not bounded or start <= datetime.fromisoformat(point['issue_time'].replace('Z', '+00:00')) < end][-limit:]
    return success_response(data)
