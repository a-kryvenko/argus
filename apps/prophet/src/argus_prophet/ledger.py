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
        self.skipped = False

    @classmethod
    def begin(cls, product, trigger, config):
        from psycopg.types.json import Jsonb
        run_id = uuid4()
        # Called only under the shared generation flock. A previous running row
        # in this same single-host workspace therefore belongs to a dead writer.
        scope = str(config.workdir.resolve())
        details = provenance(config)
        with connect() as conn:
            conn.execute("""UPDATE prophet.forecast_run SET status='interrupted', finished_at=%s,
                error='Previous writer ended without recording completion'
                WHERE scope=%s AND status='running'""", (datetime.now(UTC), scope))
            conn.execute("""INSERT INTO prophet.forecast_run
                (id,scope,product,trigger,started_at,status,provenance) VALUES (%s,%s,%s,%s,%s,'running',%s)""",
                         (run_id, scope, product, trigger, datetime.now(UTC), Jsonb(details)))
        return cls(run_id)

    def snapshot(self, inputs):
        from psycopg.types.json import Jsonb
        payload = inputs.model_dump(mode='json')
        with connect() as conn:
            result = conn.execute("""UPDATE prophet.forecast_run SET input_snapshot=%s,input_sha256=%s
                WHERE id=%s AND status='running' AND input_snapshot IS NULL""",
                                  (Jsonb(payload), fingerprint(payload), self.run_id))
            if result.rowcount != 1:
                raise RuntimeError('Run snapshot can only be recorded once')

    def store(self, name, path, model_info, row_count, columns):
        from psycopg.types.json import Jsonb
        if row_count < 1:
            raise ValueError('Cannot record an empty forecast')
        content = Path(path).read_bytes()
        with connect() as conn:
            conn.execute("""INSERT INTO prophet.forecast_artifact
                (run_id,name,status,created_at,csv_gzip,sha256,row_count,columns,model_info)
                VALUES (%s,%s,'stored',%s,%s,%s,%s,%s,%s)""",
                         (self.run_id, name, datetime.now(UTC), gzip.compress(content, mtime=0),
                          hashlib.sha256(content).hexdigest(), row_count, Jsonb(list(columns)), Jsonb(model_info)))

    def csv_written(self, name):
        with connect() as conn:
            result = conn.execute("""UPDATE prophet.forecast_artifact SET csv_written_at=%s
                WHERE run_id=%s AND name=%s AND status='stored'""", (datetime.now(UTC), self.run_id, name))
            if result.rowcount != 1:
                raise RuntimeError('CSV acknowledgement has no stored artifact')

    def skip(self, name, reason):
        from psycopg.types.json import Jsonb
        with connect() as conn:
            conn.execute("""INSERT INTO prophet.forecast_artifact
                (run_id,name,status,created_at,model_info,error) VALUES (%s,%s,'skipped',%s,%s,%s)""",
                         (self.run_id, name, datetime.now(UTC), Jsonb({}), str(reason)[:2000]))
        self.skipped = True

    def finish(self, error=None):
        status = 'failed' if error is not None else 'partial' if self.skipped else 'succeeded'
        message = f'{type(error).__name__}: {error}'[:2000] if error is not None else None
        with connect() as conn:
            result = conn.execute("""UPDATE prophet.forecast_run SET status=%s,finished_at=%s,error=%s
                WHERE id=%s AND status='running'""", (status, datetime.now(UTC), message, self.run_id))
            if result.rowcount != 1:
                raise RuntimeError("Run is no longer running")
            if error is None:
                from argus_prophet.publication import publish_run
                publish_run(conn, self.run_id)


def list_runs(limit=20):
    from psycopg.rows import dict_row
    if not 1 <= limit <= 100:
        raise ValueError('limit must be between 1 and 100')
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT id,product,trigger,started_at,finished_at,status,error
            FROM prophet.forecast_run ORDER BY started_at DESC,id DESC LIMIT %s''', (limit,))
        return cursor.fetchall()


def describe_run(run_id, include_inputs=False):
    from psycopg.rows import dict_row
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT id,product,trigger,started_at,finished_at,status,error,
            input_sha256,provenance FROM prophet.forecast_run WHERE id=%s''', (run_id,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError('Run not found')
        if include_inputs:
            cursor.execute('SELECT input_snapshot FROM prophet.forecast_run WHERE id=%s', (run_id,))
            result['input_snapshot'] = cursor.fetchone()['input_snapshot']
        cursor.execute('''SELECT name,status,created_at,csv_written_at,sha256,row_count,columns,model_info,error
            FROM prophet.forecast_artifact WHERE run_id=%s ORDER BY name''', (run_id,))
        result['artifacts'] = cursor.fetchall()
        cursor.execute('''SELECT r.id,r.product,r.published_at,r.issue_time,e.attempts,e.exported_at,e.error,
            (c.release_id IS NOT NULL) AS is_current FROM prophet.forecast_release r
            JOIN prophet.forecast_export e ON e.release_id=r.id
            LEFT JOIN prophet.current_forecast c ON c.release_id=r.id
            WHERE r.run_id=%s ORDER BY r.product''', (run_id,))
        result['releases'] = cursor.fetchall()
        return result
