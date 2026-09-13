"""Retry current published CSV exports without recalculating forecasts.

All callers must hold the shared generation lock, including publication writers.
"""
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from common.config import get_config
from argus_prophet.db.session import connect
from argus_prophet.publication import read_release

logger = logging.getLogger(__name__)


def write_csv(path, content, release_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        archive = path.parent / 'archive'
        archive.mkdir(parents=True, exist_ok=True)
        # Retried exports do not overwrite the pre-export backup.
        archived = archive / f'{path.stem}-before-{release_id}.csv'
        if not archived.exists():
            shutil.copyfile(path, archived)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_current(*, force=False):
    config = get_config()
    with connect() as conn:
        pending = conn.execute('''SELECT c.product,c.release_id FROM prophet.current_forecast c
            JOIN prophet.forecast_export e ON e.release_id=c.release_id
            WHERE (%s OR e.exported_at IS NULL) ORDER BY c.product''', (force,)).fetchall()
    failures = []
    for product, release_id in pending:
        with connect() as conn:
            conn.execute('UPDATE prophet.forecast_export SET attempts=attempts+1,exported_at=NULL,error=NULL WHERE release_id=%s', (release_id,))
        try:
            release = read_release(product, release_id)
            for artifact in release.artifacts:
                path = config.workdir / config.models_registry['models'][artifact.name]['forecast_path']
                write_csv(path, artifact.csv_text.encode('utf-8'), release_id)
            with connect() as conn:
                conn.execute('UPDATE prophet.forecast_export SET exported_at=%s,error=NULL WHERE release_id=%s',
                             (datetime.now(UTC), release_id))
                conn.execute('''UPDATE prophet.forecast_artifact SET csv_written_at=%s
                    WHERE run_id=%s AND name=ANY(%s)''',
                    (datetime.now(UTC), release.run_id, [a.name for a in release.artifacts]))
        except Exception as exc:
            logger.exception('CSV export failed for release %s', release_id)
            failures.append(product)
            with connect() as conn:
                conn.execute('UPDATE prophet.forecast_export SET error=%s WHERE release_id=%s',
                             (f'{type(exc).__name__}: {exc}'[:2000], release_id))
    if failures:
        raise RuntimeError('CSV export pending for: ' + ', '.join(failures))
    return len(pending)
