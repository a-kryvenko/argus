"""Bounded JSON transport for dashboard service reads and writes."""
import asyncio
from dataclasses import dataclass
import json
from typing import Annotated

from fastapi import Depends, Request
import httpx


class UpstreamUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Reply:
    status_code: int
    content: bytes = b''

    def json(self):
        try:
            return json.loads(self.content)
        except RecursionError:
            raise ValueError('Invalid upstream JSON') from None


def upstream_client(*, max_connections: int = 32) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=False, trust_env=False,
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections // 2),
        headers={'User-Agent': 'Argus-Dashboard/1', 'Accept-Encoding': 'identity'},
    )


async def intelligence_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.intelligence_client


async def monitoring_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.monitoring_client


IntelligenceClient = Annotated[httpx.AsyncClient, Depends(intelligence_client)]
MonitoringClient = Annotated[httpx.AsyncClient, Depends(monitoring_client)]


async def read_response(client: httpx.AsyncClient, method: str, base: str, path: str, token: str, *,
                        deadline: float, max_bytes: int, accepted: tuple[int, ...],
                        params=None, body=None) -> Reply:
    timeout = httpx.Timeout(deadline, connect=min(deadline, 10), pool=min(deadline, 2))
    try:
        async with asyncio.timeout(deadline):
            async with client.stream(
                method, base.rstrip('/') + path, follow_redirects=False, timeout=timeout,
                headers={'Authorization': f'Bearer {token}', 'Accept-Encoding': 'identity'},
                params=params, json=body,
            ) as response:
                if response.status_code not in accepted:
                    return Reply(response.status_code)
                # Response bodies are bounded before any content decompression.
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise UpstreamUnavailable
                length = response.headers.get('Content-Length')
                if length is not None and not 0 <= int(length) <= max_bytes:
                    raise UpstreamUnavailable
                content = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    if len(content) + len(chunk) > max_bytes:
                        raise UpstreamUnavailable
                    content.extend(chunk)
                return Reply(response.status_code, bytes(content))
    except (httpx.HTTPError, httpx.InvalidURL, TimeoutError, ValueError):
        raise UpstreamUnavailable from None
