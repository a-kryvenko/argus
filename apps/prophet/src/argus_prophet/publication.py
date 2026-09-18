"""Prophet-owned publication pointers. Call writers under the database generation_lock."""
import gzip
import io
from datetime import UTC, datetime
from uuid import uuid4

from common.schemas.forecast_release import (
    ForecastArtifact, ForecastRelease, MAX_ARTIFACT_BYTES, PRODUCT_ARTIFACTS,
)
from argus_prophet.db.session import connect


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


def publish_run(conn, run_id):
    """Publish complete products in the caller's run-completion transaction."""
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    with conn.cursor(row_factory=dict_row) as cursor:
        # Serializes pointer decisions, including repeatable historical promotion.
        cursor.execute("SELECT pg_advisory_xact_lock(736218, 1)")
        cursor.execute("SELECT status,started_at,product FROM prophet.forecast_run WHERE id=%s", (run_id,))
        run = cursor.fetchone()
        if run is None or run['status'] not in ('succeeded', 'partial'):
            return []
        cursor.execute("SELECT * FROM prophet.forecast_artifact WHERE run_id=%s AND status='stored'", (run_id,))
        rows = {row['name']: row for row in cursor.fetchall()}
        published = []
        products = PRODUCT_ARTIFACTS if run['product'] == 'all' else {
            run['product']: PRODUCT_ARTIFACTS[run['product']]}
        for product, names in products.items():
            if not set(names).issubset(rows):
                if run['product'] != 'all':
                    raise ValueError(f'Incomplete artifacts for {product}')
                continue
            cursor.execute('''SELECT r.issue_time, f.started_at, r.run_id FROM prophet.current_forecast c
                JOIN prophet.forecast_release r ON r.id=c.release_id
                JOIN prophet.forecast_run f ON f.id=r.run_id WHERE c.product=%s''', (product,))
            current = cursor.fetchone()
            if current and (current['run_id'] == run_id or current['started_at'] >= run['started_at']):
                continue
            artifacts = [_artifact(rows[name]) for name in names]
            release = ForecastRelease(release_id=uuid4(), run_id=run_id, product=product,
                                      published_at=datetime.now(UTC), issue_time=artifacts[0].issue_time(),
                                      artifacts=artifacts)
            if current and current['issue_time'] > release.issue_time:
                continue
            cursor.execute('''INSERT INTO prophet.forecast_release
                (id,product,run_id,published_at,issue_time,artifact_names) VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (product,run_id) DO NOTHING RETURNING id''',
                (release.release_id, product, run_id, release.published_at, release.issue_time, Jsonb(list(names))))
            inserted = cursor.fetchone()
            if inserted is None:
                continue
            cursor.execute('''INSERT INTO prophet.current_forecast(product,release_id) VALUES (%s,%s)
                ON CONFLICT(product) DO UPDATE SET release_id=excluded.release_id''', (product, release.release_id))
            published.append(release.release_id)
        return published


def read_release(product, release_id=None):
    from psycopg.rows import dict_row
    if product not in PRODUCT_ARTIFACTS:
        raise ReleaseNotFound(product)
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        if release_id is None:
            cursor.execute('''SELECT r.* FROM prophet.current_forecast c
                JOIN prophet.forecast_release r ON r.id=c.release_id WHERE c.product=%s''', (product,))
        else:
            cursor.execute('SELECT * FROM prophet.forecast_release WHERE product=%s AND id=%s', (product, release_id))
        row = cursor.fetchone()
        if row is None:
            raise ReleaseNotFound(product)
        cursor.execute('''SELECT * FROM prophet.forecast_artifact
            WHERE run_id=%s AND name=ANY(%s) AND status='stored' ORDER BY name''', (row['run_id'], row['artifact_names']))
        return ForecastRelease(release_id=row['id'], product=product, run_id=row['run_id'],
                               published_at=row['published_at'], issue_time=row['issue_time'],
                               artifacts=[_artifact(artifact) for artifact in cursor.fetchall()])
