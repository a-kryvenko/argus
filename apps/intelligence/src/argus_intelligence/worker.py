"""Poll published releases; transactionally commit the stub result and attempt."""
import json
import signal
import threading
from uuid import uuid4

from argus_intelligence import db

LOCK = (736218, 3)


def process_once(product='solar-wind-speed', *, fetch=None, assess=None):
    from argus_intelligence.cli import check
    from psycopg.types.json import Jsonb
    fetch = fetch or check
    assess = assess or (lambda evidence: evidence)
    with db.connect() as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(%s,%s) AS acquired', LOCK).fetchone()['acquired']:
            return {'status': 'busy', 'product': product}
        # The session lock excludes all Intelligence writers. Interrupted work is retryable.
        conn.execute("""UPDATE intelligence.attempt SET status='interrupted',finished_at=now(),
                     error='Writer stopped before completion' WHERE status='running'""")
        pending = conn.execute("""SELECT a.release_id FROM intelligence.attempt a
            WHERE a.product=%s AND a.release_id IS NOT NULL AND a.status IN ('failed','interrupted')
            AND NOT EXISTS (SELECT 1 FROM intelligence.result r
                WHERE r.product=a.product AND r.release_id=a.release_id)
            ORDER BY a.started_at LIMIT 1""", (product,)).fetchone()
        pinned = pending['release_id'] if pending else None
        attempt_id = uuid4()
        conn.execute("""INSERT INTO intelligence.attempt(id,product,release_id,status)
                     VALUES (%s,%s,%s,'running')""", (attempt_id, product, pinned))
        try:
            evidence = fetch(product, release_id=pinned)
            release_id = evidence['release_id']
            conn.execute("""UPDATE intelligence.attempt SET release_id=%s,prophet_run_id=%s
                         WHERE id=%s""", (release_id, evidence['run_id'], attempt_id))
            if conn.execute('SELECT 1 FROM intelligence.result WHERE product=%s AND release_id=%s',
                            (product, release_id)).fetchone():
                conn.execute("UPDATE intelligence.attempt SET status='skipped',finished_at=now() WHERE id=%s", (attempt_id,))
                return {'status': 'skipped', 'product': product, 'release_id': str(release_id)}
            result = assess(evidence)
            with conn.transaction():
                conn.execute("""INSERT INTO intelligence.result(product,release_id,prophet_run_id,attempt_id,result)
                             VALUES (%s,%s,%s,%s,%s)""",
                             (product, release_id, evidence['run_id'], attempt_id, Jsonb(result)))
                conn.execute("UPDATE intelligence.attempt SET status='succeeded',finished_at=now() WHERE id=%s", (attempt_id,))
            return {'status': 'succeeded', 'product': product, 'release_id': str(release_id), 'mode': 'stub'}
        except Exception:
            # Use this same locked session; a broken connection cannot publish or reconnect.
            if not conn.closed and not conn.broken:
                conn.execute("""UPDATE intelligence.attempt SET status='failed',finished_at=now(),
                             error='Release retrieval or stub processing failed' WHERE id=%s""", (attempt_id,))
            raise


def status(product='solar-wind-speed'):
    with db.connect() as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            latest = conn.execute('SELECT * FROM intelligence.attempt WHERE product=%s ORDER BY started_at DESC,id DESC LIMIT 1',
                                  (product,)).fetchone()
            result = conn.execute('SELECT * FROM intelligence.result WHERE product=%s ORDER BY processed_at DESC LIMIT 1',
                                  (product,)).fetchone()
            pending = conn.execute("""SELECT count(DISTINCT a.release_id) AS count FROM intelligence.attempt a
                WHERE a.product=%s AND a.status IN ('failed','interrupted','running') AND a.release_id IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM intelligence.result r WHERE r.product=a.product AND r.release_id=a.release_id)""",
                                   (product,)).fetchone()['count']
    return {'service': 'intelligence', 'mode': 'stub', 'product': product,
            'latest_attempt': latest, 'latest_result': result, 'pending_releases': pending,
            'note': 'Processing history only; this does not establish worker liveness, forecast readiness or satellite risks.'}


def worker(product='solar-wind-speed', *, interval=60, stop=None):
    stop = stop or threading.Event()
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.is_set():
            try:
                outcome = process_once(product)
            except Exception:
                outcome = {'status': 'error', 'product': product, 'error': 'Processing failed; retry scheduled'}
            print(json.dumps(outcome), flush=True)
            stop.wait(interval)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
