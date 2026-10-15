"""Owned copies of validated releases; no forecast or risk calculations."""
import gzip
import hashlib
import io
import json

from common.schemas.forecast_release import ForecastRelease
from argus_intelligence.cli import MAX_RESPONSE_BYTES


def retain(conn, run_id, inputs, *, captured_at):
    """Called within the assessment transaction. Deduplicate immutable releases."""
    for product, release in inputs.items():
        if release is None:
            continue
        if not isinstance(release, ForecastRelease) or release.product != product:
            raise ValueError('Unexpected forecast input')
        content = json.dumps(release.model_dump(mode='json'), sort_keys=True,
                             separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        if len(content) > MAX_RESPONSE_BYTES:
            raise ValueError('Forecast input too large')
        checksum = hashlib.sha256(content).hexdigest()
        stored = conn.execute('''SELECT product,sha256 FROM intelligence.forecast_input
            WHERE release_id=%s''', (release.release_id,)).fetchone()
        if stored is None:
            conn.execute('''INSERT INTO intelligence.forecast_input
                (release_id,product,sha256,content_gzip,captured_at) VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (release_id) DO NOTHING''',
                (release.release_id, product, checksum,
                 gzip.compress(content, compresslevel=3, mtime=0), captured_at))
            stored = conn.execute('''SELECT product,sha256 FROM intelligence.forecast_input
                WHERE release_id=%s''', (release.release_id,)).fetchone()
        if stored['product'] != product or stored['sha256'] != checksum:
            raise ValueError('Published release changed under the same ID')
        conn.execute('''INSERT INTO intelligence.density_run_input(run_id,product,release_id)
            VALUES (%s,%s,%s)''', (run_id, product, release.release_id))


def decode(row):
    with gzip.GzipFile(fileobj=io.BytesIO(row['content_gzip'])) as stream:
        content = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES or hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Retained forecast input is corrupt')
    release = ForecastRelease.model_validate_json(content)
    if release.release_id != row['release_id'] or release.product != row['product']:
        raise ValueError('Unexpected retained forecast input')
    return release
