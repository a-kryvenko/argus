"""PostgreSQL transaction locks for source ingestion."""
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import text

SOURCE_LOCKS = {
    'solar_wind_mag': 730100,
    'solar_wind_plasma': 730101,
    'kp': 730110,
    'dst': 730111,
}

FILE_LOCKS = {'sdo-live': 730200, 'sdo-warmup': 730201, 'sdo-browse': 730210,
              'donki': 730220, 'gong': 730230, 'goes': 730231}


_lane_locks = {}


@contextmanager
def ingestion_lock(name, number):
    """Coordinate threads locally and worker processes through PostgreSQL."""
    from clio.db import session as db
    lock = _lane_locks.setdefault(name, Lock())
    if not lock.acquire(blocking=False):
        yield False
        return
    try:
        with db.get_sync_engine().begin() as connection:
            acquired = connection.dialect.name != 'postgresql' or connection.scalar(
                text('SELECT pg_try_advisory_xact_lock(:number)'), {'number': number})
            yield acquired
    finally:
        lock.release()
