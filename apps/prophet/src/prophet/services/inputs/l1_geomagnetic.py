"""Receipt-gated native observations for the rapid geomagnetic forecast."""
from datetime import UTC, datetime, timedelta

import pandas as pd

from prophet.services.inputs.observations import _kp_start, _usable_value, read_series

MINUTE = timedelta(minutes=1)
WIND_HISTORY = timedelta(hours=4)
INDEX_HISTORY = timedelta(hours=18)
WIND_PRODUCTS = {
    'v': 'swpc.rtsw_plasma', 'n': 'swpc.rtsw_plasma',
    'bx': 'swpc.rtsw_mag', 'by': 'swpc.rtsw_mag', 'bz': 'swpc.rtsw_mag', 'bt': 'swpc.rtsw_mag',
}
INDEXES = {
    'kp': (10800, timedelta(hours=4)),
    'dst': (3600, timedelta(hours=2)),
}


def _utc(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _cutoff(as_of):
    if as_of.utcoffset() is None:
        raise ValueError('Expected a timezone-aware observation cutoff')
    return as_of.astimezone(UTC)


def _contract(payload, metric, basis, seconds):
    series = payload.get(metric, {})
    if (series.get('time_basis'), series.get('aggregation'), series.get('resolution_seconds')) != (
        basis, 'source', seconds,
    ):
        raise ValueError(f'Expected native Clio series for {metric}')
    return series['points']


def _known_points(points, start, cutoff, delay):
    for point in points:
        if not isinstance(point, dict) or not _usable_value(point):
            continue
        at = _utc(point.get('observed_at'))
        receipt = _utc(point.get('received_at'))
        if at is not None and receipt is not None and start <= at <= receipt <= cutoff and at + delay <= cutoff:
            yield at, receipt, point


def _usable_wind(metric, value):
    if metric == 'v':
        return 100 <= value <= 2500
    if metric == 'n':
        return 0 < value <= 150
    if metric == 'bt':
        return 0 <= value < 400
    return abs(value) < 400


def _usable_index(metric, start, value):
    if metric == 'kp':
        return _kp_start(start) and 0 <= value <= 9
    return not (start.minute or start.second or start.microsecond) and -2000 <= value <= 1000


def read_l1_wind_history(client, url, token, as_of):
    """Keep closed L1 vectors and standalone plasma speeds for the travel clock."""
    cutoff = _cutoff(as_of)
    end = cutoff.replace(second=0, microsecond=0)
    start = end - WIND_HISTORY
    payload = read_series(client.get, url, token, WIND_PRODUCTS, interval='1m', time_basis='l1',
                          **{'from': start.isoformat(), 'to': end.isoformat()})
    clock = pd.date_range(start, end, inclusive='left', freq='min', name='l1_time')
    rows = []
    for metric, product in WIND_PRODUCTS.items():
        points = _contract(payload, metric, 'l1', 60)
        for at, receipt, point in _known_points(points, start, cutoff, MINUTE):
            spacecraft = point.get('spacecraft')
            value = point['value']
            if (at.second or at.microsecond or point.get('source_product') != product
                    or not isinstance(spacecraft, str) or not spacecraft.strip() or not _usable_wind(metric, value)):
                continue
            rows.append((at, metric, float(value), spacecraft.strip().casefold(), receipt))
    samples = pd.DataFrame(rows, columns=['l1_time', 'metric', 'value', 'spacecraft', 'received_at'])
    samples['l1_time'] = pd.to_datetime(samples.l1_time, utc=True)
    samples['received_at'] = pd.to_datetime(samples.received_at, utc=True)
    grouped = samples.groupby('l1_time')
    values = samples.pivot(index='l1_time', columns='metric', values='value')
    wind = values.reindex(index=clock, columns=list(WIND_PRODUCTS)).astype(float)
    coherent = wind.notna().all(axis=1) & grouped.spacecraft.nunique().reindex(clock, fill_value=0).eq(1)
    wind = wind.where(coherent, axis=0).rename_axis(columns=None)
    wind['received_at'] = grouped.received_at.max().reindex(clock).where(coherent)
    wind['spacecraft'] = grouped.spacecraft.first().reindex(clock).where(coherent)
    plasma = samples.loc[samples.metric.eq('v')].set_index('l1_time')
    wind['plasma_speed'] = plasma['value'].reindex(clock).astype(float)
    wind['plasma_received_at'] = plasma['received_at'].reindex(clock)
    return wind


def read_l1_geomagnetic_indices(client, url, token, as_of):
    """Keep received Kp and Dst at the interval starts used by training."""
    cutoff = _cutoff(as_of)
    start = cutoff - INDEX_HISTORY
    payload = read_series(client.get, url, token, INDEXES, interval='1m', time_basis='l1',
                          **{'from': start.isoformat(), 'to': cutoff.isoformat()})
    history = {}
    for metric, (seconds, delay) in INDEXES.items():
        points = _contract(payload, metric, 'source_interval_start', seconds)
        known = {
            at: point['value'] for at, _, point in _known_points(points, start, cutoff, delay)
            if _usable_index(metric, at, point['value'])
        }
        clock = pd.DatetimeIndex(known.keys(), tz='UTC', name='time')
        history[metric] = pd.Series(list(known.values()), index=clock, dtype=float, name=metric).sort_index()
    return history['kp'], history['dst']
