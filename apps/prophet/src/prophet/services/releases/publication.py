"""Prophet-owned publication pointers. Call writers under the database generation_lock."""
import asyncio
import gzip
import io
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from common.schemas.forecast_release import (
    ForecastArtifact, ForecastRelease, MAX_ARTIFACT_BYTES, PRODUCT_ARTIFACTS,
)
from prophet.db.session import get_db_session


class ReleaseNotFound(LookupError):
    pass


def _artifact(row):
    # Bound decompression even if the stored archive is corrupt.
    with gzip.GzipFile(fileobj=io.BytesIO(row['csv_gzip'])) as stream:
        content = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError('Forecast artifact exceeds the read contract size limit')
    return ForecastArtifact(name=row['name'], sha256=row['sha256'], row_count=row['row_count'],
                            columns=row['columns'], csv_text=content.decode('utf-8'), model_info=row['model_info'])


async def publish_run(session, run_id):
    """Publish one complete product in the caller's run-completion transaction."""
    # Serializes pointer decisions, including repeatable historical promotion.
    await session.execute(text("SELECT pg_advisory_xact_lock(736218, 1)"))
    query = await session.execute(text("SELECT status,started_at,product FROM forecast_run WHERE id=:run_id"), {'run_id': run_id})
    run = query.mappings().first()
    if run is None or run['status'] != 'succeeded' or run['product'] == 'all':
        return None
    query = await session.execute(text("SELECT * FROM forecast_artifact WHERE run_id=:run_id AND status='stored'"), {'run_id': run_id})
    rows = {row['name']: row for row in query.mappings().all()}
    product = run['product']
    names = PRODUCT_ARTIFACTS[product]
    if not set(names).issubset(rows):
        raise ValueError(f'Incomplete artifacts for {product}')
    query = await session.execute(text('''SELECT r.issue_time, f.started_at, r.run_id FROM current_forecast c
        JOIN forecast_release r ON r.id=c.release_id
        JOIN forecast_run f ON f.id=r.run_id WHERE c.product=:product'''), {'product': product})
    current = query.mappings().first()
    if current and (current['run_id'] == run_id or current['started_at'] >= run['started_at']):
        return None
    release = await asyncio.to_thread(_release_from_artifacts, {
        'id': uuid4(), 'run_id': run_id, 'product': product, 'published_at': datetime.now(UTC),
    }, [rows[name] for name in names])
    if current and current['issue_time'] > release.issue_time:
        return None
    query = await session.execute(text('''INSERT INTO forecast_release
        (id,product,run_id,published_at,issue_time,artifact_names) VALUES (:release_id,:product,:run_id,:published_at,:issue_time,:artifact_names)
        ON CONFLICT (product,run_id) DO NOTHING RETURNING id''').bindparams(bindparam('artifact_names', type_=JSONB)), {'release_id': release.release_id, 'product': product, 'run_id': run_id, 'published_at': release.published_at, 'issue_time': release.issue_time, 'artifact_names': list(names)})
    inserted = query.mappings().first()
    if inserted is None:
        return None
    await session.execute(text('''INSERT INTO current_forecast(product,release_id) VALUES (:product,:release_id)
        ON CONFLICT(product) DO UPDATE SET release_id=excluded.release_id'''), {'product': product, 'release_id': release.release_id})
    return release.release_id


async def read_release(product, release_id=None):
    if product not in PRODUCT_ARTIFACTS:
        raise ReleaseNotFound(product)
    async with get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        if release_id is None:
            query = await session.execute(text('''SELECT r.* FROM current_forecast c
                JOIN forecast_release r ON r.id=c.release_id WHERE c.product=:product'''), {'product': product})
        else:
            query = await session.execute(text('SELECT * FROM forecast_release WHERE product=:product AND id=:release_id'), {'product': product, 'release_id': release_id})
        row = query.mappings().first()
        if row is None:
            raise ReleaseNotFound(product)
        query = await session.execute(text('''SELECT * FROM forecast_artifact
            WHERE run_id=:run_id AND name=ANY(:artifact_names) AND status='stored' ORDER BY name'''), {'run_id': row['run_id'], 'artifact_names': row['artifact_names']})
        artifacts = query.mappings().all()
    return await asyncio.to_thread(_release_from_artifacts, row, artifacts)


def _release_from_artifacts(row, artifacts):
    artifacts = [_artifact(artifact) for artifact in artifacts]
    return ForecastRelease(release_id=row['id'], product=row['product'], run_id=row['run_id'],
                           published_at=row['published_at'],
                           issue_time=row.get('issue_time') or artifacts[0].issue_time(), artifacts=artifacts)
