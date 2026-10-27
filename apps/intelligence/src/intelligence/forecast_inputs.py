"""Owned copies of validated releases; no forecast or risk calculations."""
import gzip
import hashlib
import io
import json

from sqlalchemy import text

from common.schemas.forecast_release import ForecastRelease
from intelligence.cli import MAX_RESPONSE_BYTES


async def retain(session, run_id, inputs, *, captured_at):
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
        stored = (await session.execute(text('''SELECT product,sha256 FROM forecast_input
            WHERE release_id=:release_id'''), {'release_id': release.release_id})).mappings().first()
        if stored is None:
            await session.execute(text('''INSERT INTO forecast_input
                (release_id,product,sha256,content_gzip,captured_at) VALUES (:release_id,:product,:sha256,:content_gzip,:captured_at)
                ON CONFLICT (release_id) DO NOTHING'''), {'release_id': release.release_id, 'product': product, 'sha256': checksum, 'content_gzip': gzip.compress(content, compresslevel=3, mtime=0), 'captured_at': captured_at})
            stored = (await session.execute(text('''SELECT product,sha256 FROM forecast_input
                WHERE release_id=:release_id'''), {'release_id': release.release_id})).mappings().first()
        if stored['product'] != product or stored['sha256'] != checksum:
            raise ValueError('Published release changed under the same ID')
        await session.execute(text('''INSERT INTO density_run_input(run_id,product,release_id)
            VALUES (:run_id,:product,:release_id)'''), {'run_id': run_id, 'product': product, 'release_id': release.release_id})


def decode(row):
    with gzip.GzipFile(fileobj=io.BytesIO(row['content_gzip'])) as stream:
        content = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES or hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Retained forecast input is corrupt')
    release = ForecastRelease.model_validate_json(content)
    if release.release_id != row['release_id'] or release.product != row['product']:
        raise ValueError('Unexpected retained forecast input')
    return release
