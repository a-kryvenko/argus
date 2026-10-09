"""Durable product schedules and Prophet writer locks coordinated by PostgreSQL."""
from contextlib import contextmanager
from datetime import UTC, datetime

from argus_prophet.db.session import open_connection, transaction

# Distinct from the publication transaction lock (736218, 1) and Clio job locks.
GENERATION_LOCK = (736218, 2)
VERIFICATION_LOCK = (736218, 3)
COMPLETED = ('succeeded', 'imported')


class GenerationBusy(RuntimeError):
    """Another Prophet writer owns the database session lock."""


def recover_interrupted(writer):
    # Owning the global writer lock means no previous DB writer remains active.
    with transaction(writer) as conn:
        conn.execute("""UPDATE prophet.forecast_run SET status='interrupted',finished_at=%s,
            error='Previous writer ended without recording completion' WHERE status='running'""", (datetime.now(UTC),))
        conn.execute("""UPDATE prophet.forecast_slot SET status='interrupted',finished_at=%s,
            error='Previous writer ended without recording completion' WHERE status='running'""", (datetime.now(UTC),))


@contextmanager
def writer_lock(key, *, recover=False):
    # Closing this unpooled session releases its lock, including on exceptions.
    # It stays in autocommit between short operations, not idle in a long transaction.
    with open_connection(autocommit=True) as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(%s,%s)', key).fetchone()[0]:
            raise GenerationBusy('Another Prophet task owns this writer lock')
        if recover:
            recover_interrupted(conn)
        yield conn


def generation_lock():
    return writer_lock(GENERATION_LOCK, recover=True)


@contextmanager
def verification_lock():
    # Same acquisition order as retention; scoring must not overlap inference.
    from common.config import get_config
    from argus_prophet.services.heavy_task import heavy_task
    with writer_lock(GENERATION_LOCK), writer_lock(VERIFICATION_LOCK) as writer:
        with heavy_task(get_config().data_root):
            yield writer


@contextmanager
def retention_lock():
    # Fixed order. Verification has independent writes but must not race deletion
    # of the releases it is scoring. Reuse the generation writer connection.
    with generation_lock() as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(%s,%s)', VERIFICATION_LOCK).fetchone()[0]:
            raise GenerationBusy('Verification is running; retry cleanup later')
        try:
            yield conn
        finally:
            if not conn.closed and not conn.broken:
                conn.execute('SELECT pg_advisory_unlock(%s,%s)', VERIFICATION_LOCK)


def product_pending(product, slot, *, writer):
    """A product's successes never suppress another product or a later slot."""
    with transaction(writer) as conn:
        completed = conn.execute('''SELECT max(slot) FROM prophet.forecast_slot
            WHERE product=ANY(%s) AND status=ANY(%s)''', ([product, 'all'], list(COMPLETED))).fetchone()[0]
        if completed is not None and completed >= slot:
            return False
        return product not in completed_products(conn, slot)


def completed_products(conn, slot) -> set[str]:
    """Durable per-product success, including releases from historical batch runs."""
    rows = conn.execute("""SELECT product FROM prophet.forecast_run
        WHERE scheduled_slot=%s AND status='succeeded' AND product<>'all'
        UNION
        SELECT r.product FROM prophet.forecast_release r
        JOIN prophet.forecast_run f ON f.id=r.run_id
        WHERE f.scheduled_slot=%s AND f.product='all' AND f.status IN ('succeeded','partial')""",
        (slot, slot)).fetchall()
    return {row[0] for row in rows}
