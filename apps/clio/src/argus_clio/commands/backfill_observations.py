"""Backfill missing measurements and rebuild hourly observations in a UTC range."""
import json

from argus_clio.db.session import get_session_factory
from argus_clio.services.observations.backfill import backfill



async def run(args):
    async with get_session_factory()() as session:
        result = await backfill(session, args.start, args.end)
    print(json.dumps(result, indent=2))
