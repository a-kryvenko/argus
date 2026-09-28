"""Native NOAA estimated Kp and Kyoto real-time Dst, without gap filling."""
from datetime import UTC, datetime, timedelta
import math

import requests

SOURCES = {
    'kp': 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
    'dst': 'https://services.swpc.noaa.gov/products/kyoto-dst.json',
}
INTERVAL_SECONDS = {'kp': 10800, 'dst': 3600}
POLL_SECONDS = {'kp': 60, 'dst': 300}


def parse_records(metric: str, payload: object, received_at: datetime) -> list[dict]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f'{metric}: expected a nonempty JSON record list')
    field = 'Kp' if metric == 'kp' else 'dst'
    records = {}
    for row in payload:
        if not isinstance(row, dict) or not {'time_tag', field} <= row.keys():
            raise ValueError(f'{metric}: unexpected source schema')
        try:
            start = datetime.fromisoformat(row['time_tag'].replace('Z', '+00:00'))
            start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f'{metric}: invalid time_tag') from exc
        if start.minute or start.second or start.microsecond or (metric == 'kp' and start.hour % 3):
            raise ValueError(f'{metric}: unexpected interval alignment')
        if start > received_at:
            continue
        try:
            value = float(row[field]) if row[field] is not None and not isinstance(row[field], bool) else None
        except (TypeError, ValueError):
            value = None
        if value is not None and (not math.isfinite(value) or abs(value) >= 9999 or (metric == 'kp' and not 0 <= value <= 9)):
            value = None
        quality = 'missing' if value is None else 'unverified'
        if value is not None and metric == 'kp' and row.get('station_count') == 0:
            quality = 'flagged'
        records[start] = {'metric': metric, 'interval_start': start,
                          'interval_end': start + timedelta(seconds=INTERVAL_SECONDS[metric]),
                          'value': value, 'quality': quality, 'received_at': received_at, 'raw': row}
    if not records:
        raise ValueError(f'{metric}: no usable timestamps')
    return list(records.values())


def fetch_records(metric: str) -> list[dict]:
    response = requests.get(SOURCES[metric], timeout=(5, 25))
    response.raise_for_status()
    return parse_records(metric, response.json(), datetime.now(UTC))
