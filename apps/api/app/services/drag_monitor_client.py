"""Bounded, authenticated transport for the private spacecraft monitor."""
import os

import httpx
from fastapi import HTTPException
from pydantic import TypeAdapter

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def request(method, owner_id, suffix, response_type, *, body=None, params=None, transport=None, resource='drag-monitor', max_response_bytes=None):
    max_response_bytes = MAX_RESPONSE_BYTES if max_response_bytes is None else max_response_bytes
    url, token = os.getenv('INTELLIGENCE_URL'), os.getenv('INTELLIGENCE_SERVICE_TOKEN')
    if not url or not token:
        raise HTTPException(503, 'Drag monitoring is not configured')
    try:
        with httpx.Client(timeout=httpx.Timeout(180, connect=10), follow_redirects=False,
                          trust_env=False, transport=transport) as client:
            with client.stream(method, url.rstrip('/') + f'/internal/v1/{resource}/{owner_id}' + suffix,
                               headers={'Authorization': f'Bearer {token}'}, params=params,
                               json=body.model_dump(mode='json') if body else None) as response:
                if response.status_code in (404, 409):
                    raise HTTPException(response.status_code, 'Spacecraft not found' if response.status_code == 404
                                        else 'Profile changed, paused or assessment running. Reload and try again.')
                if response.status_code == 422 and resource == 'density-advisory':
                    raise HTTPException(422, 'Invalid trajectory or reference. Check timestamps, validity and ratio thresholds.')
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > max_response_bytes:
                        raise ValueError('Monitoring response too large')
        return TypeAdapter(response_type).validate_json(content)
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, 'Density advisory is unavailable' if resource == 'density-advisory' else 'Drag monitoring is unavailable') from None
