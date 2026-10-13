"""Official public DONKI API; bounded responses, no credentials or HTML scraping."""
import json
from datetime import datetime
import requests

BASE_URL = 'https://ccmc.gsfc.nasa.gov/DONKI-API/get/'
ENDPOINTS = ('CME', 'WSAEnlilSimulations', 'HSS', 'IPS')
MAX_BYTES = 8 * 1024 * 1024


def invalid_constant(value):
    raise ValueError('Nonfinite DONKI value: ' + value)


def decode(content):
    if len(content) > MAX_BYTES:
        raise ValueError('DONKI response exceeds 8 MiB')
    payload = json.loads(content, parse_constant=invalid_constant)
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError('DONKI response must be a list of objects')
    return payload


def validate(endpoint, payload):
    identity, time_field = {
        'CME': ('activityID', 'startTime'),
        'HSS': ('hssID', 'eventTime'),
        'IPS': ('activityID', 'eventTime'),
        'WSAEnlilSimulations': ('simulationID', 'modelCompletionTime'),
    }[endpoint]
    for row in payload:
        if not isinstance(row.get(identity), str) or not row[identity]:
            raise ValueError('DONKI record identity is missing')
        value = row.get(time_field)
        if not isinstance(value, str):
            raise ValueError('DONKI record time is missing')
        time = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if time.tzinfo is None:
            raise ValueError('DONKI record time must include a timezone')
    return payload


def fetch(endpoint, start, end):
    if endpoint not in ENDPOINTS:
        raise ValueError('Unknown DONKI endpoint')
    with requests.get(BASE_URL + endpoint,
                      params={'startDate': start.date().isoformat(), 'endDate': end.date().isoformat()},
                      timeout=(10, 45), stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('DONKI did not return HTTP 200')
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('DONKI response exceeds 8 MiB')
            chunks.append(chunk)
        raw = b''.join(chunks)
    return raw, validate(endpoint, decode(raw))
