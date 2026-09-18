"""Execution evidence and transactional publication of completed product releases."""
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from argus_prophet.db.session import connect

logger = logging.getLogger(__name__)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def provenance(config):
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    sources = {}
    for name in ('argus_prophet', 'forecast', 'forecast_core', 'common'):
        spec = importlib.util.find_spec(name)
        if spec and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            sources[name] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(root.rglob('*.py'))}
    return {'python': platform.python_version(), 'packages': packages, 'source_sha256': sources,
            'models_registry': config.models_registry}


class RunRecorder:
    def __init__(self, run_id):
        self.run_id = run_id

    @classmethod
    def begin(cls, product, trigger, config, *, scheduled_slot=None, details=None):
        from psycopg.types.json import Jsonb
        from argus_prophet.products import PRODUCTS
        if product not in PRODUCTS:
            raise ValueError('Expected one supported forecast product')
        run_id = uuid4()
        if (trigger == 'scheduled') != (scheduled_slot is not None):
            raise ValueError('Scheduled attempts require a slot; manual attempts must not have one')
        scope = str(config.workdir.resolve())
        details = provenance(config) if details is None else details
        with connect(writing=True) as conn:
            if scheduled_slot is not None:
                result = conn.execute("""INSERT INTO prophet.forecast_slot(slot,status,attempts,started_at)
                    VALUES (%s,'running',1,%s) ON CONFLICT(slot) DO UPDATE
                    SET status='running',attempts=forecast_slot.attempts+1,
                        started_at=EXCLUDED.started_at,finished_at=NULL,error=NULL
                    WHERE forecast_slot.status IN ('failed','partial','interrupted') RETURNING slot""",
                    (scheduled_slot, datetime.now(UTC)))
                if result.fetchone() is None:
                    raise RuntimeError('Scheduled slot is already active or complete')
            conn.execute("""INSERT INTO prophet.forecast_run
                (id,scope,product,trigger,started_at,status,provenance,scheduled_slot)
                VALUES (%s,%s,%s,%s,%s,'running',%s,%s)""",
                (run_id, scope, product, trigger, datetime.now(UTC), Jsonb(details), scheduled_slot))
        return cls(run_id)

    def snapshot(self, inputs):
        from psycopg.types.json import Jsonb
        payload = inputs.model_dump(mode='json')
        from argus_prophet.readiness import input_diagnostics
        diagnostics = input_diagnostics(inputs)
        with connect(writing=True) as conn:
            result = conn.execute("""UPDATE prophet.forecast_run SET input_snapshot=%s,input_sha256=%s,
                provenance=jsonb_set(provenance,'{input_diagnostics}',%s)
                WHERE id=%s AND status='running' AND input_snapshot IS NULL""",
                                  (Jsonb(payload), fingerprint(payload), Jsonb(diagnostics), self.run_id))
            if result.rowcount != 1:
                raise RuntimeError('Run snapshot can only be recorded once')

    def store(self, name, content: bytes, model_info, row_count, columns):
        if row_count < 1:
            raise ValueError('Cannot record an empty forecast')
        from psycopg.types.json import Jsonb
        compressed = gzip.compress(content, mtime=0)
        digest = hashlib.sha256(content).hexdigest()
        with connect(writing=True) as conn:
            conn.execute("""INSERT INTO prophet.forecast_artifact
                (run_id,name,status,created_at,csv_gzip,sha256,row_count,columns,model_info)
                VALUES (%s,%s,'stored',%s,%s,%s,%s,%s,%s)""",
                         (self.run_id, name, datetime.now(UTC), compressed,
                          digest, row_count, Jsonb(list(columns)), Jsonb(model_info)))

    def finish(self, error=None):
        status = 'failed' if error is not None else 'succeeded'
        message = f'{type(error).__name__}: {error}'[:2000] if error is not None else None
        with connect(writing=True) as conn:
            result = conn.execute("""UPDATE prophet.forecast_run SET status=%s,finished_at=%s,error=%s
                WHERE id=%s AND status='running' RETURNING scheduled_slot""", (status, datetime.now(UTC), message, self.run_id))
            if result.rowcount != 1:
                raise RuntimeError("Run is no longer running")
            if error is None:
                from argus_prophet.publication import publish_run
                publish_run(conn, self.run_id)
            slot = result.fetchone()[0]
            if slot is not None:
                from argus_prophet.worker import completed_products
                from argus_prophet.products import PRODUCTS
                completed = completed_products(conn, slot)
                pending = set(PRODUCTS) - completed
                slot_status = 'succeeded' if not pending else 'partial' if completed else 'failed'
                slot_error = 'Pending products: ' + ', '.join(sorted(pending)) if pending else None
                changed = conn.execute("""UPDATE prophet.forecast_slot SET status=%s,finished_at=%s,error=%s
                    WHERE slot=%s AND status='running'""", (slot_status, datetime.now(UTC), slot_error, slot))
                if changed.rowcount != 1:
                    raise RuntimeError('Scheduled slot is no longer active')


def list_runs(limit=20):
    from psycopg.rows import dict_row
    if not 1 <= limit <= 100:
        raise ValueError('limit must be between 1 and 100')
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT id,product,trigger,scheduled_slot,started_at,finished_at,status,error
            FROM prophet.forecast_run ORDER BY started_at DESC,id DESC LIMIT %s''', (limit,))
        return cursor.fetchall()


def describe_run(run_id, include_inputs=False):
    from psycopg.rows import dict_row
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT id,product,trigger,scheduled_slot,started_at,finished_at,status,error,
            input_sha256,provenance FROM prophet.forecast_run WHERE id=%s''', (run_id,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError('Run not found')
        if include_inputs:
            cursor.execute('SELECT input_snapshot FROM prophet.forecast_run WHERE id=%s', (run_id,))
            result['input_snapshot'] = cursor.fetchone()['input_snapshot']
        cursor.execute('''SELECT name,status,created_at,sha256,row_count,columns,model_info,error
            FROM prophet.forecast_artifact WHERE run_id=%s ORDER BY name''', (run_id,))
        result['artifacts'] = cursor.fetchall()
        cursor.execute('''SELECT r.id,r.product,r.published_at,r.issue_time,
            (c.release_id IS NOT NULL) AS is_current FROM prophet.forecast_release r
            LEFT JOIN prophet.current_forecast c ON c.release_id=r.id
            WHERE r.run_id=%s ORDER BY r.product''', (run_id,))
        result['releases'] = cursor.fetchall()
        return result
