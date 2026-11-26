"""Authenticated requests and profile responses from Intelligence."""
import os

import httpx
from fastapi import HTTPException
from pydantic import TypeAdapter

from api.services.upstream import Reply, UpstreamUnavailable, read_response

PROFILE_LIMIT_BYTES = 8 * 1024 * 1024
# Leave time to return a response before the dashboard proxy times out at 120 seconds.
PROFILE_DEADLINE_SECONDS = 110


async def request(client: httpx.AsyncClient, method, path, *, body=None, params=None, validation=False) -> Reply:
    url, token = os.getenv('INTELLIGENCE_URL'), os.getenv('INTELLIGENCE_SERVICE_TOKEN')
    if not url or not token:
        raise HTTPException(503, 'Intelligence service is not configured')
    try:
        return await read_response(
            client, method, url, path, token, deadline=PROFILE_DEADLINE_SECONDS, max_bytes=PROFILE_LIMIT_BYTES,
            accepted=(200, 201, 422) if validation else (200, 201), params=params,
            body=body.model_dump(mode='json') if body is not None else None,
        )
    except UpstreamUnavailable:
        raise HTTPException(503, 'Intelligence service is unavailable') from None


def profile_response(response: Reply, response_type):
    if response.status_code in (404, 409):
        raise HTTPException(response.status_code, 'Spacecraft not found' if response.status_code == 404
                            else 'Profile changed, paused or assessment running. Reload and try again.')
    if response.status_code not in (200, 201):
        raise HTTPException(503, 'Intelligence service is unavailable')
    try:
        return TypeAdapter(response_type).validate_json(response.content)
    except ValueError:
        raise HTTPException(503, 'Intelligence service is unavailable') from None
