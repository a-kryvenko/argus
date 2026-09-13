"""Read observations via Clio. Never fall back to SQL or provider downloads."""
import os
from datetime import datetime

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

READS = frozenset({
    'latest', 'history', 'status', 'summary', 'solar-wind/latest',
    'solar-wind/history', 'geomagnetic/latest', 'geomagnetic/history', 'browse',
})


async def read_observations(resource: str, params: dict | None = None):
    if resource not in READS:
        raise ValueError('Unsupported Clio read contract')
    url = os.getenv('OBSERVATIONS_URL')
    token = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise HTTPException(503, 'Observation service is not configured')
    query = {key: value.isoformat() if isinstance(value, datetime) else value
             for key, value in (params or {}).items() if value is not None}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10),
                                     follow_redirects=False, trust_env=False) as client:
            response = await client.get(
                url.rstrip('/') + '/internal/v1/observations/' + resource,
                params=query, headers={'Authorization': f'Bearer {token}'},
            )
        if response.status_code == 422:
            return JSONResponse(status_code=422, content=response.json())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('success'), bool):
            raise ValueError('Invalid observation response')
        return payload
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, 'Observation service is unavailable') from None
