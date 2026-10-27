"""Prophet writer exclusion and transactions on the lock-owning connection."""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from prophet.db.session import get_db_session

# Distinct from the publication transaction lock (736218, 1) and Clio job locks.
GENERATION_LOCK = (736218, 2)
VERIFICATION_LOCK = (736218, 3)


class GenerationBusy(RuntimeError):
    """Another Prophet writer owns the database session lock."""


@dataclass
class Writer:
    session: AsyncSession
    connection: AsyncConnection
    lost: bool = False


@asynccontextmanager
async def transaction(writer: Writer | None) -> AsyncGenerator[AsyncSession, None]:
    """Commit on the lock-owning connection; a lost writer cannot reconnect."""
    if writer is None:
        raise RuntimeError('Prophet writes require a database writer lock')
    if writer.lost or writer.connection.closed or writer.connection.invalidated:
        raise RuntimeError('Prophet lock connection was lost; reacquire the lock for a new attempt')
    try:
        async with writer.session.begin():
            yield writer.session
    finally:
        if writer.connection.closed or writer.connection.invalidated:
            writer.lost = True


async def recover_interrupted(writer):
    # Owning the global writer lock means no previous DB writer remains active.
    async with transaction(writer) as session:
        await session.execute(text("""UPDATE prophet.forecast_run SET status='interrupted',finished_at=:now,
            error='Previous writer ended without recording completion' WHERE status='running'"""),
            {'now': datetime.now(UTC)})


@asynccontextmanager
async def writer_lock(key, *, recover=False) -> AsyncGenerator[Writer, None]:
    async with get_db_session() as session:
        connection = await session.connection()
        try:
            acquired = await session.scalar(text('SELECT pg_try_advisory_lock(:namespace,:key)'),
                                            {'namespace': key[0], 'key': key[1]})
            await session.commit()
        except BaseException:
            # Acquisition may have reached PostgreSQL before cancellation or an I/O error.
            await connection.invalidate()
            raise
        if not acquired:
            raise GenerationBusy('Another Prophet task owns this writer lock')
        writer = Writer(session, connection)
        try:
            if recover:
                await recover_interrupted(writer)
            yield writer
        finally:
            # Session locks survive commits. Never return a locked connection to the pool.
            if not writer.lost and not connection.closed and not connection.invalidated:
                try:
                    await session.rollback()
                    await session.execute(text('SELECT pg_advisory_unlock(:namespace,:key)'),
                                          {'namespace': key[0], 'key': key[1]})
                    await session.commit()
                except BaseException:
                    await connection.invalidate()
                    raise


def generation_lock():
    return writer_lock(GENERATION_LOCK, recover=True)


@asynccontextmanager
async def verification_lock() -> AsyncGenerator[Writer, None]:
    from common.config import get_config
    from prophet.services.heavy_task import heavy_task
    # Same acquisition order as retention; scoring must not overlap inference.
    async with writer_lock(GENERATION_LOCK), writer_lock(VERIFICATION_LOCK) as writer:
        with heavy_task(get_config().data_root):
            yield writer


@asynccontextmanager
async def retention_lock() -> AsyncGenerator[Writer, None]:
    async with generation_lock() as writer, writer_lock(VERIFICATION_LOCK):
        yield writer
