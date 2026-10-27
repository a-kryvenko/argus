"""Poll published releases; transactionally commit the stub result and attempt."""
import asyncio
import json
import os
import signal
import threading
from uuid import uuid4

from common.db.session import dispose_engine
from intelligence.db import session as db
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

LOCK = (736218, 3)


async def process_once(product='solar-wind-speed', *, fetch=None, assess=None):
    from intelligence.cli import check
    fetch = fetch or check
    assess = assess or (lambda evidence: evidence)
    attempt_id = None
    try:
        async with db.get_db_session() as session, session.begin():
            acquired = await session.scalar(text('SELECT pg_try_advisory_xact_lock(:namespace,:key)'),
                                            {'namespace': LOCK[0], 'key': LOCK[1]})
            if not acquired:
                return {'status': 'busy', 'product': product}
            # Journal commits survive a crash while the processing transaction holds the lock.
            async with db.get_db_session() as journal, journal.begin():
                await journal.execute(text("""UPDATE intelligence.attempt SET status='interrupted',finished_at=now(),
                    error='Writer stopped before completion' WHERE status='running'"""))
                pinned = await journal.scalar(text("""SELECT a.release_id FROM intelligence.attempt a
                    WHERE a.product=:product AND a.release_id IS NOT NULL AND a.status IN ('failed','interrupted')
                    AND NOT EXISTS (SELECT 1 FROM intelligence.result r
                        WHERE r.product=a.product AND r.release_id=a.release_id)
                    ORDER BY a.started_at LIMIT 1"""), {'product': product})
                attempt_id = uuid4()
                await journal.execute(text("""INSERT INTO intelligence.attempt(id,product,release_id,status)
                    VALUES (:id,:product,:release_id,'running')"""),
                    {'id': attempt_id, 'product': product, 'release_id': pinned})
            evidence = await asyncio.to_thread(fetch, product, release_id=pinned)
            release_id = evidence['release_id']
            async with db.get_db_session() as journal, journal.begin():
                await journal.execute(text("""UPDATE intelligence.attempt SET release_id=:release_id,
                    prophet_run_id=:run_id WHERE id=:id"""),
                    {'release_id': release_id, 'run_id': evidence['run_id'], 'id': attempt_id})
            if await session.scalar(text('SELECT 1 FROM intelligence.result WHERE product=:product AND release_id=:release_id'),
                                    {'product': product, 'release_id': release_id}):
                await session.execute(text("UPDATE intelligence.attempt SET status='skipped',finished_at=clock_timestamp() WHERE id=:id"),
                                      {'id': attempt_id})
                return {'status': 'skipped', 'product': product, 'release_id': str(release_id)}
            result = await asyncio.to_thread(assess, evidence)
            await session.execute(text("""INSERT INTO intelligence.result(product,release_id,prophet_run_id,attempt_id,result,processed_at)
                VALUES (:product,:release_id,:run_id,:attempt_id,:result,clock_timestamp())""").bindparams(bindparam('result', type_=JSONB)),
                {'product': product, 'release_id': release_id, 'run_id': evidence['run_id'],
                 'attempt_id': attempt_id, 'result': result})
            await session.execute(text("UPDATE intelligence.attempt SET status='succeeded',finished_at=clock_timestamp() WHERE id=:id"),
                                  {'id': attempt_id})
        return {'status': 'succeeded', 'product': product, 'release_id': str(release_id), 'mode': 'stub'}
    except Exception:
        if attempt_id is not None:
            async with db.get_db_session() as journal, journal.begin():
                await journal.execute(text("""UPDATE intelligence.attempt SET status='failed',finished_at=now(),
                    error='Release retrieval or stub processing failed' WHERE id=:id AND status='running'"""),
                    {'id': attempt_id})
        raise


async def status(product='solar-wind-speed'):
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        latest = (await session.execute(text('SELECT * FROM intelligence.attempt WHERE product=:product ORDER BY started_at DESC,id DESC LIMIT 1'),
                                        {'product': product})).mappings().first()
        result = (await session.execute(text('SELECT * FROM intelligence.result WHERE product=:product ORDER BY processed_at DESC LIMIT 1'),
                                        {'product': product})).mappings().first()
        pending = await session.scalar(text("""SELECT count(DISTINCT a.release_id) FROM intelligence.attempt a
            WHERE a.product=:product AND a.status IN ('failed','interrupted','running') AND a.release_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM intelligence.result r WHERE r.product=a.product AND r.release_id=a.release_id)"""),
                                       {'product': product})
    return {'service': 'intelligence', 'mode': 'stub', 'product': product,
            'latest_attempt': dict(latest) if latest else None, 'latest_result': dict(result) if result else None,
            'pending_releases': pending,
            'note': 'Processing history only; this does not establish worker liveness, forecast readiness or satellite risks.'}


async def _worker(product='solar-wind-speed', *, interval=60, stop=None):
    stop = stop or threading.Event()
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.is_set():
            try:
                outcome = await process_once(product)
            except Exception:
                outcome = {'status': 'error', 'product': product, 'error': 'Processing failed; retry scheduled'}
            print(json.dumps(outcome), flush=True)
            if os.getenv('INTELLIGENCE_EXPERIMENTAL_DRAG_MONITOR','false').lower() in ('true','1'):
                try:
                    from intelligence.monitor import refresh_profiles
                    monitoring = await refresh_profiles(stop=stop)
                    if monitoring['profiles']:
                        print(json.dumps({'drag_monitor': monitoring}), flush=True)
                except Exception:
                    print(json.dumps({'drag_monitor': {'error': 'Monitoring unavailable; retry scheduled'}}), flush=True)
            try:
                from intelligence.density_monitor import refresh_profiles as refresh_density
                screening = await refresh_density(stop=stop)
                if screening['profiles']:
                    print(json.dumps({'density_advisory': screening}), flush=True)
            except Exception:
                print(json.dumps({'density_advisory': {'error': 'Monitoring unavailable; retry scheduled'}}), flush=True)
            await asyncio.to_thread(stop.wait, interval)
    finally:
        try:
            await dispose_engine()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def worker(product='solar-wind-speed', *, interval=60, stop=None):
    asyncio.run(_worker(product, interval=interval, stop=stop))
