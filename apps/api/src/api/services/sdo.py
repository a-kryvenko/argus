"""Reuse Clio connections across concurrent image requests."""
import httpx

_client = None


def image_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10),
                                   follow_redirects=False, trust_env=False)
    return _client


async def close_image_client():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
