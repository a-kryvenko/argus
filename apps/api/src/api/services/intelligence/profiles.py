"""Authenticated requests and profile responses from Intelligence."""
import os

import httpx
from fastapi import HTTPException
from pydantic import TypeAdapter


def request(method, path, *, body=None, params=None, transport=None):
    url, token = os.getenv('INTELLIGENCE_URL'), os.getenv('INTELLIGENCE_SERVICE_TOKEN')
    if not url or not token:
        raise HTTPException(503, 'Intelligence service is not configured')
    try:
        with httpx.Client(timeout=httpx.Timeout(180, connect=10), follow_redirects=False,
                          trust_env=False, transport=transport) as client:
            return client.request(method, url.rstrip('/') + path,
                headers={'Authorization': f'Bearer {token}'}, params=params,
                json=body.model_dump(mode='json') if body is not None else None)
    except httpx.HTTPError:
        raise HTTPException(503, 'Intelligence service is unavailable') from None


def profile_response(response, response_type):
    if response.status_code in (404, 409):
        raise HTTPException(response.status_code, 'Spacecraft not found' if response.status_code == 404
                            else 'Profile changed, paused or assessment running. Reload and try again.')
    try:
        response.raise_for_status()
        return TypeAdapter(response_type).validate_json(response.content)
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, 'Intelligence service is unavailable') from None
