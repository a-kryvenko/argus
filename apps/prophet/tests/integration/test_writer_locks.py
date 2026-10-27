"""Writer exclusion, lost connections and atomic publication for one-shot commands."""
import asyncio
from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from tests.support.database import database, runtime
from .storage import recorder_database, store_product
from prophet.services.releases.recording import RunRecorder
from prophet.services.releases.publication import read_release, ReleaseNotFound
from prophet.db.locks import generation_lock, GenerationBusy, transaction

pytestmark = pytest.mark.anyio


async def test_independent_sessions_cannot_generate_concurrently(recorder_database):
    async with generation_lock() as writer:
        async def contender():
            async with generation_lock():
                pytest.fail('Competing writer acquired the lock')
        with pytest.raises(GenerationBusy):
            await asyncio.wait_for(asyncio.create_task(contender()), 10)
    async with generation_lock() as writer:
        async with transaction(writer) as session:
            assert await session.scalar(text('SELECT 1')) == 1


async def test_disconnected_writer_cannot_publish_after_new_owner(recorder_database):
    dsn, passwords, config = recorder_database
    async with generation_lock() as writer:
        old = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
        (await store_product(old))
        async with transaction(writer) as session:
            pid = await session.scalar(text('SELECT pg_backend_pid()'))
        with psycopg.connect(dsn['prophet'], autocommit=True) as admin:
            assert admin.execute('SELECT pg_terminate_backend(%s)', (pid,)).fetchone()[0]
        with pytest.raises((SQLAlchemyError, RuntimeError)):
            (await old.finish())
        # Even a new writer in the same thread cannot rebind the old recorder.
        async with generation_lock() as replacement:
            fresh = await RunRecorder.begin('dst', 'manual', config, writer=replacement)
            await store_product(fresh)
            await fresh.finish()
        with pytest.raises((SQLAlchemyError, RuntimeError)):
            (await old.finish())
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (old.run_id,)).fetchone()[0] == 'interrupted'
        assert conn.execute('SELECT product FROM prophet.forecast_release ORDER BY product').fetchall() == [
            ('dst',)]
    assert (await read_release('dst')).run_id != old.run_id


async def test_publication_and_product_completion_roll_back_together(recorder_database):
    dsn, passwords, config = recorder_database
    async with generation_lock() as writer:
        run = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
        (await store_product(run))
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute('UPDATE prophet.forecast_artifact SET sha256=%s', ('0' * 64,))
        with pytest.raises(ValueError, match='checksum'):
            (await run.finish())
        with pytest.raises(ReleaseNotFound):
            (await read_release('dst'))
        with runtime(dsn, 'prophet', passwords) as conn:
            assert conn.execute('SELECT status FROM prophet.forecast_run').fetchone()[0] == 'running'
    async with generation_lock() as replacement:
        fresh = await RunRecorder.begin('dst', 'manual', config, writer=replacement)
        await store_product(fresh)
        await fresh.finish()
    assert (await read_release('dst')).run_id != run.run_id


async def test_verification_excludes_generation_and_cleanup(recorder_database):
    from prophet.db.locks import verification_lock, retention_lock
    acquired, release = asyncio.Event(), asyncio.Event()
    async def verify():
        async with verification_lock() as writer:
            async with transaction(writer) as session:
                await session.execute(text('SELECT 1'))
            acquired.set()
            await asyncio.wait_for(release.wait(), 10)
    task = asyncio.create_task(verify())
    try:
        await asyncio.wait_for(acquired.wait(), 10)
        with pytest.raises(GenerationBusy):
            async with generation_lock():
                pytest.fail('Generation overlapped verification')
        with pytest.raises(GenerationBusy):
            async with retention_lock():
                pytest.fail('Cleanup overlapped verification')
    finally:
        release.set()
        await task
    async with retention_lock() as writer:
        async with transaction(writer) as session:
            await session.execute(text('SELECT 1'))


async def test_generation_excludes_verification_and_releases_locks_after_failure(recorder_database):
    from prophet.db.locks import verification_lock, retention_lock
    with pytest.raises(ValueError, match='calculation failed'):
        async with generation_lock():
            with pytest.raises(GenerationBusy):
                async with verification_lock():
                    pytest.fail('Verification overlapped generation')
            raise ValueError('calculation failed')
    with pytest.raises(ValueError, match='verification failed'):
        async with verification_lock() as writer:
            async with transaction(writer) as session:
                await session.execute(text('SELECT 1'))
            raise ValueError('verification failed')
    # Neither the failed acquisition nor either task failure leaks a lock.
    async with generation_lock() as writer:
        async with transaction(writer) as session:
            await session.execute(text('SELECT 1'))
    async with verification_lock() as writer:
        async with transaction(writer) as session:
            await session.execute(text('SELECT 1'))
    async with retention_lock() as writer:
        async with transaction(writer) as session:
            await session.execute(text('SELECT 1'))


async def test_legacy_batch_is_readable_but_cannot_be_finished_or_republished(recorder_database):
    from prophet.services.releases.publication import publish_run
    from prophet.db.locks import transaction
    dsn, passwords, config = recorder_database
    async with generation_lock() as writer:
        with pytest.raises(ValueError, match='one supported'):
            (await RunRecorder.begin('all', 'manual', config, writer=writer))
        run = (await RunRecorder.begin('dst', 'manual', config, writer=writer))
        (await store_product(run))
        (await run.finish())
        release = (await read_release('dst'))
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute("UPDATE prophet.forecast_run SET product='all' WHERE id=%s", (run.run_id,))
        async with transaction(writer) as conn:
            assert (await publish_run(conn, run.run_id)) is None
        assert (await read_release('dst')) == release
        with runtime(dsn, 'prophet', passwords) as conn:
            conn.execute("UPDATE prophet.forecast_run SET status='running' WHERE id=%s", (run.run_id,))
        with pytest.raises(RuntimeError):
            (await run.finish())
        with runtime(dsn, 'prophet', passwords) as conn:
            assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s',
                                (run.run_id,)).fetchone()[0] == 'running'
        assert (await read_release('dst')) == release


async def test_cron_runs_publish_without_scheduler_slots_and_recover_crashes(recorder_database):
    dsn, passwords, config = recorder_database
    async with generation_lock() as writer:
        completed = await RunRecorder.begin('dst', 'scheduled', config, writer=writer)
        await store_product(completed)
        await completed.finish()
        interrupted = await RunRecorder.begin('hmf', 'scheduled', config, writer=writer)
    release = await read_release('dst')
    async with generation_lock():
        pass
    assert await read_release('dst') == release
    with runtime(dsn, 'prophet', passwords) as conn:
        assert conn.execute('SELECT trigger,status,scheduled_slot,scheduled_product FROM prophet.forecast_run WHERE id=%s',
                            (completed.run_id,)).fetchone() == ('scheduled', 'succeeded', None, None)
        assert conn.execute('SELECT status FROM prophet.forecast_run WHERE id=%s', (interrupted.run_id,)).fetchone()[0] == 'interrupted'
        assert conn.execute('SELECT count(*) FROM prophet.forecast_slot').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM prophet.scheduled_job').fetchone()[0] == 0
