"""Observed trends only: explicit coverage and no interpolation across gaps."""
from datetime import UTC, datetime, timedelta
import math
from statistics import mean

from sqlalchemy.ext.asyncio import AsyncSession
from argus_clio.services import solar_wind, geomagnetic

MIN_COVERAGE = 0.8
LOOKBACK_MINUTES = 75


def valid(point: dict) -> bool:
    value = point['value']
    return value is not None and math.isfinite(value) and point['quality'] not in ('missing', 'flagged')


def unavailable(reason: str, **extra) -> dict:
    return {'status': 'unavailable', 'value': None, 'reason': reason, **extra}


def change_one_hour(points: list[dict], now: datetime) -> dict:
    if not points or not valid(points[-1]):
        return unavailable('missing_latest')
    last = points[-1]
    end = last['observed_at']
    if (now-end).total_seconds() > solar_wind.STALE_AFTER_SECONDS:
        return unavailable('stale', as_of=end)
    by_time = {point['observed_at']: point for point in points if valid(point)}
    windows = [[by_time.get(end-timedelta(minutes=minute)) for minute in range(a, b)]
               for a, b in [(0, 5), (60, 65), (0, 60)]]
    coverage = [sum(point is not None for point in window)/len(window) for window in windows]
    info = {'as_of': end, 'coverage': {'recent': coverage[0], 'previous': coverage[1], 'last_hour': coverage[2]},
            'minimum_coverage': MIN_COVERAGE, 'method': 'difference_of_5_minute_means_60_minutes_apart'}
    if min(coverage) < MIN_COVERAGE:
        return unavailable('insufficient_coverage', **info)
    relevant = [point for point in points if end-timedelta(minutes=64) <= point['observed_at'] <= end]
    if any(point['spacecraft'] != last['spacecraft'] for point in relevant):
        return unavailable('source_changed', **info)
    recent, previous = (mean(point['value'] for point in window if point is not None) for window in windows[:2])
    return {'status': 'available', 'value': round(recent-previous, 3),
            'recent_mean': recent, 'previous_mean': previous, **info}


def southward_duration(points: list[dict], now: datetime) -> dict:
    if not points or not valid(points[-1]):
        return unavailable('missing_latest')
    last = points[-1]
    end = last['observed_at']
    if (now-end).total_seconds() > solar_wind.STALE_AFTER_SECONDS:
        return unavailable('stale', as_of=end)
    if last['value'] >= 0:
        return {'status': 'available', 'value': 0, 'unit': 'sampled_minutes', 'as_of': end}
    expected = end
    count = 0
    for point in reversed(points):
        if point['observed_at'] != expected or not valid(point):
            return unavailable('insufficient_coverage', as_of=end)
        if point['spacecraft'] != last['spacecraft']:
            return unavailable('source_changed', as_of=end)
        if point['value'] >= 0:
            return {'status': 'available', 'value': count, 'unit': 'sampled_minutes', 'as_of': end}
        count += 1
        expected -= timedelta(minutes=1)
    # All available samples are consecutive and negative; the onset is outside the window.
    return {'status': 'lower_bound', 'value': count, 'unit': 'sampled_minutes',
            'reason': 'onset_before_available_window', 'as_of': end}


async def summary(session: AsyncSession, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    current = await solar_wind.latest(session, list(solar_wind.METADATA), now)
    indices = await geomagnetic.latest(session, now)
    past = await solar_wind.history(session, ['v', 'n', 'bz', 'bt'],
                                    now-timedelta(minutes=LOOKBACK_MINUTES), now+timedelta(microseconds=1))
    changes = {metric: {**change_one_hour(series['points'], now), 'unit': series['unit']}
               for metric, series in past['series'].items()}
    return {'generated_at': now, 'solar_wind': current['series'], 'geomagnetic': indices['series'],
            'changes_1h': changes, 'southward_bz': southward_duration(past['series']['bz']['points'], now),
            'lookback_minutes': LOOKBACK_MINUTES}
