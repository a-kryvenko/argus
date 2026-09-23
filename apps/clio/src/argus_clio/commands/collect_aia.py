"""Collect every hourly AIA193 slot, retaining original FITS and receipt time."""
import argparse,asyncio
from argus_clio.services.aia import collect_aia
from argus_clio.db.session import dispose_engine


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history-days',type=int,default=40);args=parser.parse_args()
    async def run():
        try:await collect_aia(history_days=args.history_days)
        finally:await dispose_engine()
    asyncio.run(run())
