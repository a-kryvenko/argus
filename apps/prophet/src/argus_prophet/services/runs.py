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

from argus_prophet.db.session import transaction

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
            'models_registry': config.models_registry,
            'prophet_config': getattr(config, 'project_config', {}).get('prophet', {})}


class RunRecorder:
    def __init__(self, run_id, writer):
        self.run_id = run_id
        self.writer = writer

    @classmethod
    def begin(cls, product, trigger, config, *, writer, scheduled_slot=None, details=None):
        from psycopg.types.json import Jsonb
        from argus_prophet.services.generation.products import PRODUCTS
        if product not in PRODUCTS:
            raise ValueError('Expected one supported forecast product')
        run_id = uuid4()
        if (trigger == 'scheduled') != (scheduled_slot is not None):
            raise ValueError('Scheduled attempts require a slot; manual attempts must not have one')
        scope = str(config.workdir.resolve())
        details = provenance(config) if details is None else details
        with transaction(writer) as conn:
            if scheduled_slot is not None:
                result = conn.execute("""INSERT INTO prophet.forecast_slot(product,slot,status,attempts,started_at)
                    VALUES (%s,%s,'running',1,%s) ON CONFLICT(product,slot) DO UPDATE
                    SET status='running',attempts=forecast_slot.attempts+1,
                        started_at=EXCLUDED.started_at,finished_at=NULL,error=NULL
                    WHERE forecast_slot.status IN ('failed','partial','interrupted') RETURNING slot""",
                    (product, scheduled_slot, datetime.now(UTC)))
                if result.fetchone() is None:
                    raise RuntimeError('Scheduled slot is already active or complete')
            conn.execute("""INSERT INTO prophet.forecast_run
                (id,scope,product,trigger,started_at,status,provenance,scheduled_slot,scheduled_product)
                VALUES (%s,%s,%s,%s,%s,'running',%s,%s,%s)""",
                (run_id, scope, product, trigger, datetime.now(UTC), Jsonb(details), scheduled_slot,
                 product if scheduled_slot is not None else None))
        return cls(run_id, writer)

    def snapshot(self, inputs):
        from psycopg.types.json import Jsonb
        payload = inputs.model_dump(mode='json')
        from argus_prophet.services.releases.status import input_diagnostics
        diagnostics = input_diagnostics(inputs)
        with transaction(self.writer) as conn:
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
        with transaction(self.writer) as conn:
            conn.execute("""INSERT INTO prophet.forecast_artifact
                (run_id,name,status,created_at,csv_gzip,sha256,row_count,columns,model_info)
                VALUES (%s,%s,'stored',%s,%s,%s,%s,%s,%s)""",
                         (self.run_id, name, datetime.now(UTC), compressed,
                          digest, row_count, Jsonb(list(columns)), Jsonb(model_info)))

    def finish(self, error=None):
        status = ('succeeded' if error is None else
                  'failed' if isinstance(error, Exception) else 'interrupted')
        message = f'{type(error).__name__}: {error}'[:2000] if error is not None else None
        with transaction(self.writer) as conn:
            result = conn.execute("""UPDATE prophet.forecast_run SET status=%s,finished_at=%s,error=%s
                WHERE id=%s AND status='running' AND product<>'all' RETURNING scheduled_slot,scheduled_product""", (status, datetime.now(UTC), message, self.run_id))
            if result.rowcount != 1:
                raise RuntimeError("Run is no longer running")
            if error is None:
                from argus_prophet.services.releases.publication import publish_run
                publish_run(conn, self.run_id)
            slot, slot_product = result.fetchone()
            if slot is not None:
                changed = conn.execute("""UPDATE prophet.forecast_slot SET status=%s,finished_at=%s,error=%s
                    WHERE product=%s AND slot=%s AND status='running'""",
                    (status, datetime.now(UTC), message, slot_product, slot))
                if changed.rowcount != 1:
                    raise RuntimeError('Scheduled product slot is no longer active')
