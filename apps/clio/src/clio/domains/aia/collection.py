"""Configured AIA collection: immutable observations and recoverable file artifacts."""
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.aia_snapshot import AIASnapshot
from clio.db.session import get_session_factory
from clio.domains.aia.archive import archive_root
from clio.providers.aia import fetch_snapshot, snapshot_paths

DOWNLOAD_WORKERS = 4


class AIASynopticAdapter:
    product = 'aia.synoptic_193'

    def fetch(self, slot, root, expected=None):
        return fetch_snapshot(slot, root, expected=expected, source_product=self.product)


class AIANRTAdapter(AIASynopticAdapter):
    product = 'aia.nrt_193'


def file_adapters():
    return {'aia.synoptic_193': AIASynopticAdapter(), 'aia.nrt_193': AIANRTAdapter()}


def snapshot_values(result: dict, root: Path) -> dict:
    row = dict(result)
    for key in ('raw_path', 'cache_path'):
        row[key] = str(Path(row[key]).relative_to(root.resolve()))
    for key in ('slot_at', 'observed_at', 'available_at'):
        if isinstance(row[key], str):
            row[key] = datetime.fromisoformat(row[key])
    return row


def expected_receipt(row, root):
    record = {column: getattr(row, column) for column in (
        'slot_at', 'observed_at', 'available_at', 'sha256', 'b0_deg', 'valid_fraction', 'carrington_lon')}
    for key in ('raw_path', 'cache_path'):
        record[key] = str((root / getattr(row, key)).resolve())
    record['source_product'] = getattr(row, 'source_product', None)
    return record


def _fetch_slot(slot, root, expected, sources, adapters):
    attempts = []
    raw, cache, receipt = snapshot_paths(slot, root)
    retained = expected is not None and raw.is_file() and cache.is_file()
    for product in sources:
        try:
            record = adapters[product].fetch(slot, root, expected)
            if record is not None:
                attempts.append({'product': product, 'status': 'available'})
                return dict(slot=slot, record=record, product=product,
                            outcome='retained' if retained else 'restored' if expected is not None else 'received',
                            attempts=attempts)
            # QC rejection is an observed, retained original, not an empty slot
            # to be replaced from another provider.
            if receipt.exists() and json.loads(receipt.read_text()).get('rejected'):
                attempts.append({'product': product, 'status': 'rejected'})
                return dict(slot=slot, outcome='rejected', attempts=attempts)
            attempts.append({'product': product, 'status': 'missing'})
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            attempts.append({'product': product, 'status': 'error', 'error': str(exc)})
    return dict(slot=slot, outcome='failed' if any(a['status'] == 'error' for a in attempts) else 'missing',
                attempts=attempts)


async def collect_slots(slots, sources, *, root=None, adapters=None):
    """File locks inside the adapter coordinate live/backfill at slot granularity."""
    root = (root or archive_root()).resolve()
    adapters = adapters or file_adapters()
    counts = dict(received=0, restored=0, retained=0, rejected=0, missing=0, failed=0)
    report = {**counts, 'latest_observed_at': None, 'source_errors': [], 'source_attempts': []}
    if not slots:
        return report
    async with get_session_factory()() as session:
        records = (await session.execute(select(AIASnapshot).where(
            AIASnapshot.slot_at >= min(slots), AIASnapshot.slot_at <= max(slots),
        ))).scalars().all()
    existing = {row.slot_at: expected_receipt(row, root) for row in records}
    latest = None
    statistics = {source: {'product': source, 'available': 0, 'missing': 0, 'rejected': 0, 'error': 0} for source in sources}
    ordered = sorted(set(slots), reverse=True)
    for offset in range(0, len(ordered), DOWNLOAD_WORKERS):
        outcomes = await asyncio.gather(*(asyncio.to_thread(
            _fetch_slot, slot, root, existing.get(slot), sources, adapters,
        ) for slot in ordered[offset:offset + DOWNLOAD_WORKERS]))
        inserts = []
        for result in outcomes:
            outcome = result['outcome']
            report[outcome] += 1
            for attempt in result['attempts']:
                statistics[attempt['product']][attempt['status']] += 1
                if attempt['status'] == 'error' and len(report['source_errors']) < 20:
                    report['source_errors'].append({**attempt, 'slot_at': result['slot'].isoformat()})
            if 'record' not in result:
                continue
            record = result['record']
            observed = record['observed_at']
            observed = datetime.fromisoformat(observed) if isinstance(observed, str) else observed
            latest = max(latest, observed) if latest is not None else observed
            if result['slot'] not in existing:
                inserts.append({**snapshot_values(record, root), 'source_product': record.get('source_product') or result['product']})
        if inserts:
            async with get_session_factory()() as session:
                await session.execute(insert(AIASnapshot).values(inserts).on_conflict_do_nothing(index_elements=['slot_at']))
                await session.commit()
    report['latest_observed_at'] = latest.isoformat() if latest else None
    # Keep diagnostics bounded even for a long warmup.
    report['source_attempts'] = list(statistics.values())
    return report


def observation_slots(mode, now, days, *, start=None, end=None):
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    now = now.astimezone(UTC)
    current = now.replace(minute=0, second=0, microsecond=0)
    if mode == 'live':
        return [current - timedelta(hours=h) for h in range(4)]
    if mode != 'backfill' or (start is None) != (end is None):
        raise ValueError('Use live/backfill and specify both range boundaries')
    if start is None:
        start, end = current - timedelta(days=days), current
    if (start.tzinfo is None or end.tzinfo is None or start >= end or end > now
            or any(t.minute or t.second or t.microsecond for t in (start.astimezone(UTC), end.astimezone(UTC)))):
        raise ValueError('Choose a past range of aware whole UTC hours')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if end - start > timedelta(days=60):
        raise ValueError('File backfill is bounded to 60 days')
    return [start + timedelta(hours=h) for h in range(int((end - start).total_seconds() // 3600))]


async def collect_file_observations(config, metrics, *, mode, now=None, start=None, end=None):
    now = now or datetime.now(UTC)
    files, failed = {}, []
    for metric in metrics:
        if metric != 'aia193' or metric not in config.observations:
            raise ValueError(f'Unknown configured file observation: {metric}')
        policy = config.observations[metric]
        slots = observation_slots(mode, now, policy.backfill.days, start=start, end=end)
        sources = policy.sources.live if mode == 'live' else policy.sources.historical
        result = await collect_slots(slots, sources)
        files[metric] = result
        latest = datetime.fromisoformat(result['latest_observed_at']) if result['latest_observed_at'] else None
        if (mode == 'live' and (latest is None or latest < now - timedelta(hours=3))) or (mode == 'backfill' and result['failed']):
            failed.append(metric)
    partial = failed or any(result['missing'] or result['failed'] or result['rejected'] for result in files.values())
    return {'status': 'partial' if partial else 'complete', 'failed_metrics': failed, 'files': files}

