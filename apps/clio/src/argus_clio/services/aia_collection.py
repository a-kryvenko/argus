"""Collect hourly AIA originals and persist their actual receipt metadata."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from argus_clio.db.models.aia_snapshot import AIASnapshot
from argus_clio.db.session import get_session_factory
from argus_clio.services.aia_archive import archive_root
from clio.dataloaders.aia import fetch_snapshot, hourly_slots

logger = logging.getLogger(__name__)
DOWNLOAD_WORKERS = 4


def snapshot_values(result: dict, root: Path) -> dict:
    """Convert provider receipts to database values without changing availability."""
    row = dict(result)
    for key in ('raw_path', 'cache_path'):
        row[key] = str(Path(row[key]).relative_to(root.resolve()))
    for key in ('slot_at', 'observed_at', 'available_at'):
        row[key] = datetime.fromisoformat(row[key])
    return row


async def collect_aia(*, now: datetime | None = None, history_days: int = 40) -> dict:
    slots = hourly_slots(now or datetime.now(UTC), history_days)
    async with get_session_factory()() as session:
        records = (await session.execute(select(AIASnapshot).where(
            AIASnapshot.slot_at >= slots[0], AIASnapshot.slot_at <= slots[-1],
        ))).scalars().all()

    root = archive_root()
    existing = {
        row.slot_at for row in records
        if (root / row.raw_path).is_file() and (root / row.cache_path).is_file()
    }
    # Recent images first: the forecast need not wait for historical warmup.
    pending = [slot for slot in reversed(slots) if slot not in existing]
    received = missing = failed = 0
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for begin in range(0, len(pending), DOWNLOAD_WORKERS):
            batch = pending[begin:begin + DOWNLOAD_WORKERS]
            outcomes = await asyncio.gather(*(
                loop.run_in_executor(pool, fetch_snapshot, slot, root) for slot in batch
            ), return_exceptions=True)
            async with get_session_factory()() as session:
                for result in outcomes:
                    if isinstance(result, BaseException):
                        failed += 1
                        logger.error('AIA collection failed: %s', result)
                        continue
                    if result is None:
                        missing += 1
                        continue
                    await session.execute(
                        insert(AIASnapshot).values(**snapshot_values(result, root))
                        .on_conflict_do_nothing(index_elements=['slot_at'])
                    )
                    received += 1
                await session.commit()

    logger.info(
        'AIA hourly archive: received=%s unavailable=%s failed=%s retained=%s',
        received, missing, failed, len(existing),
    )
    if failed:
        raise RuntimeError(f'{failed} AIA snapshots failed; retry next schedule')
    return dict(received=received, missing=missing, retained=len(existing))
