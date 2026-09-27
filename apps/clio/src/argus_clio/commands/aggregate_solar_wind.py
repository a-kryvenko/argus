"""Process durable five-minute/hourly aggregation backlog without fetching sources."""
import logging
from argus_clio.services.solar_wind.aggregation import aggregate_pending


async def run(args):
    count = await aggregate_pending(args.limit)
    logging.info('Aggregated %d source hours', count)
