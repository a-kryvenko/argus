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

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from prophet.db.locks import transaction

logger = logging.getLogger(__name__)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def provenance(config):
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    sources = {}
    for name in ('prophet', 'forecast', 'forecast_core', 'common'):
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
    async def begin(cls, product, trigger, config, *, writer, details=None):
        from prophet.services.generation.products import PRODUCTS
        if product not in PRODUCTS:
            raise ValueError('Expected one supported forecast product')
        run_id = uuid4()
        if trigger not in ('manual', 'scheduled'):
            raise ValueError('Unknown run trigger')
        scope = str(config.workdir.resolve())
        details = provenance(config) if details is None else details
        async with transaction(writer) as session:
            await session.execute(text("""INSERT INTO prophet.forecast_run
                (id,scope,product,trigger,started_at,status,provenance)
                VALUES (:run_id,:scope,:product,:trigger,:now,'running',:provenance)""").bindparams(bindparam('provenance', type_=JSONB)), {'run_id': run_id, 'scope': scope, 'product': product, 'trigger': trigger, 'now': datetime.now(UTC), 'provenance': details})
        return cls(run_id, writer)

    async def snapshot(self, inputs):
        payload = inputs.model_dump(mode='json')
        from prophet.services.releases.status import input_diagnostics
        diagnostics = input_diagnostics(inputs)
        async with transaction(self.writer) as session:
            result = await session.execute(text("""UPDATE prophet.forecast_run SET input_snapshot=:payload,input_sha256=:input_sha256,
                provenance=jsonb_set(provenance,'{input_diagnostics}',:input_diagnostics)
                WHERE id=:run_id AND status='running' AND input_snapshot IS NULL""").bindparams(bindparam('payload', type_=JSONB), bindparam('input_diagnostics', type_=JSONB)), {'payload': payload, 'input_sha256': fingerprint(payload), 'input_diagnostics': diagnostics, 'run_id': self.run_id})
            if result.rowcount != 1:
                raise RuntimeError('Run snapshot can only be recorded once')

    async def store(self, name, content: bytes, model_info, row_count, columns):
        if row_count < 1:
            raise ValueError('Cannot record an empty forecast')

        compressed = gzip.compress(content, mtime=0)
        digest = hashlib.sha256(content).hexdigest()
        async with transaction(self.writer) as session:
            await session.execute(text("""INSERT INTO prophet.forecast_artifact
                (run_id,name,status,created_at,csv_gzip,sha256,row_count,columns,model_info)
                VALUES (:run_id,:name,'stored',:now,:csv_gzip,:sha256,:row_count,:columns,:model_info)""").bindparams(bindparam('columns', type_=JSONB), bindparam('model_info', type_=JSONB)), {'run_id': self.run_id, 'name': name, 'now': datetime.now(UTC), 'csv_gzip': compressed, 'sha256': digest, 'row_count': row_count, 'columns': list(columns), 'model_info': model_info})

    async def finish(self, error=None):
        status = ('succeeded' if error is None else
                  'failed' if isinstance(error, Exception) else 'interrupted')
        message = f'{type(error).__name__}: {error}'[:2000] if error is not None else None
        async with transaction(self.writer) as session:
            result = await session.execute(text("""UPDATE prophet.forecast_run SET status=:status,finished_at=:now,error=:error
                WHERE id=:run_id AND status='running' AND product<>'all' RETURNING id"""), {'status': status, 'now': datetime.now(UTC), 'error': message, 'run_id': self.run_id})
            if result.rowcount != 1:
                raise RuntimeError("Run is no longer running")
            if error is None:
                from prophet.services.releases.publication import publish_run
                await publish_run(session, self.run_id)
