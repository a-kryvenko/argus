"""Observed trends only: explicit coverage and no interpolation across gaps."""
import math
from datetime import UTC, datetime, timedelta
from statistics import mean

from sqlalchemy.ext.asyncio import AsyncSession

from clio.observations import geomagnetic
from clio.observations import solar_wind as solar_wind
from clio.monitoring.specs import WIND_STALE_AFTER_SECONDS

LOOKBACK_MINUTES = 75


def valid(point: dict) -> bool:
    value = point['value']
    return value is not None and math.isfinite(value) and point['quality'] not in ('missing', 'flagged')


def unavailable(reason: str, **extra) -> dict:
    return {
        'status': 'unavailable',
        'value': None,
        'reason': reason,
        **extra
    }


def change_one_hour(points: list[dict], now: datetime) -> dict:
    if not points or not valid(points[-1]):
        return unavailable('missing_latest')
    last = points[-1]
    end = last['observed_at']

    if (now-end).total_seconds() > WIND_STALE_AFTER_SECONDS:
        return unavailable('stale', as_of=end)
    
    windows = [[point for point in points if valid(point)
                and end-timedelta(minutes=b) < point['observed_at'] <= end-timedelta(minutes=a)]
               for a, b in [(0, 5), (60, 65)]]
    coverage = [len(window)/5 for window in windows]

    info = {
        'as_of': end,
        'coverage': {
            'recent': coverage[0],
            'previous': coverage[1]
        },
        'method': 'difference_of_5_minute_means_60_minutes_apart'
    }
    
    if any(not window for window in windows):
        return unavailable('missing_comparison', **info)
    
    recent, previous = (mean(point['value'] for point in window if point is not None) for window in windows)
    return {
        'status': 'available',
        'value': round(recent-previous, 3),
        'recent_mean': recent,
        'previous_mean': previous,
        **info
    }


def value_change(current: dict | None, previous: dict | None) -> float | None:
    if current is None or previous is None or not valid(current) or not valid(previous):
        return None
    return round(current['value'] - previous['value'], 3)


def interval_change(points: list[dict], now: datetime) -> dict:
    if not points:
        return unavailable('missing_latest')
    current = points[-1]
    previous = next((point for point in reversed(points)
                     if point['interval_start'] <= now-timedelta(hours=1)), None)
    change = value_change(current, previous)
    if change is None:
        return unavailable('missing_comparison')
    return {'status': 'available', 'value': change, 'as_of': current['interval_end']}


async def summary(session: AsyncSession, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    current = await solar_wind.latest(session, list(solar_wind.METADATA), now)
    indices = await geomagnetic.latest(session, now)
    past = await solar_wind.history(
        session,
        list(solar_wind.METADATA),
        now-timedelta(minutes=LOOKBACK_MINUTES),
        now+timedelta(microseconds=1)
    )

    changes = {
        metric: {
            **change_one_hour(series['points'], now),
            'unit': series['unit']
        } for metric, series in past['series'].items()
    }

    index_history = await geomagnetic.history(
        session, now-timedelta(hours=4), now+timedelta(microseconds=1), now,
    )
    for metric, series in index_history['series'].items():
        changes[metric] = {
            **(interval_change(series['points'], now) if indices['series'][metric]['status'] != 'stale'
               else unavailable('stale')),
            'unit': series['unit'],
        }

    return {
        'generated_at': now,
        'solar_wind': current['series'],
        'geomagnetic': indices['series'],
        'changes_1h': changes,
        'lookback_minutes': LOOKBACK_MINUTES
    }
