"""Collect one batch of native solar wind samples; the worker owns repetition."""
from clio.domains.solar_wind.observations import refresh_solar_wind


async def run(args) -> None:
    await refresh_solar_wind()
