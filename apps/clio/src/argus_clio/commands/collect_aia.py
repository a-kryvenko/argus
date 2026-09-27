"""Collect every hourly AIA193 slot, retaining original FITS and receipt time."""

from argus_clio.services.aia.collection import collect_aia


async def run(args) -> None:
    await collect_aia(history_days=args.history_days)
