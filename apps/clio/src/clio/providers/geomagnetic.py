"""Native NOAA estimated Kp and Kyoto real-time Dst, without gap filling."""
from datetime import UTC, datetime
import math

import pandas as pd
import requests

from clio.providers._contract import dataset

SOURCES = {
    'kp': 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
    'dst': 'https://services.swpc.noaa.gov/products/kyoto-dst.json',
}


def parse_records(metric: str, payload: object, received_at: datetime) -> pd.DataFrame:
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
        values = {metric: value}
        if metric == 'kp':
            try:
                ap = row.get('a_running')
                values['ap'] = float(ap) if ap is not None and not isinstance(ap, bool) else None
            except (ValueError, TypeError):
                values['ap'] = None
            if row.get('station_count') == 0:
                values = dict.fromkeys(values)
        records[start] = {'timestamp': start, **values}
    return dataset(pd.DataFrame(records.values()), ('kp', 'ap') if metric == 'kp' else ('dst',))


def fetch_records(metric: str) -> pd.DataFrame:
    response = requests.get(SOURCES[metric], timeout=(5, 25))
    response.raise_for_status()
    return parse_records(metric, response.json(), datetime.now(UTC))
