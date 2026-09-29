"""Clio-owned immutable GONG inputs. Reads never import private features."""
import asyncio
import gzip
import hashlib
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from common.schemas.forecast_inputs import GONGFeatureFrame
from clio.db.models.gong_snapshot import GONGSnapshot
from clio.db.session import get_session_factory


def snapshot_values(observed, url, source):
    from clio.providers import gong as provider
    from forecast_core.observations import extract_gong_features
    content = provider.download(url)
    available = datetime.now(UTC)
    frame = GONGFeatureFrame(observed_at=observed, available_at=available,
        sha256=hashlib.sha256(content).hexdigest(), source_product=source,
        features=extract_gong_features(content))
    if not frame.features:
        raise ValueError('GONG extraction returned no features')
    return {**frame.model_dump(), 'slot_at': observed.replace(minute=0, second=0, microsecond=0),
            'source_url': url, 'fits_gzip': gzip.compress(content, mtime=0)}


async def load_gong_features(session, as_of):
    row = (await session.scalars(select(GONGSnapshot).where(
        GONGSnapshot.observed_at <= as_of,
        GONGSnapshot.available_at <= as_of,
    ).order_by(GONGSnapshot.observed_at.desc()).limit(1))).first()
    return GONGFeatureFrame.model_validate(row, from_attributes=True) if row else None


async def collect(config, *, mode, now=None, start=None, end=None):
    from clio.providers import gong as provider
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    policy = config.observations['gong']
    historical = mode == 'backfill'
    if mode not in ('live', 'backfill') or (start is None) != (end is None):
        raise ValueError('Use live/backfill and both range boundaries')
    start = (start or now - timedelta(days=policy.backfill.days)) if historical else now - timedelta(hours=3)
    end = end or now
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now or end-start > timedelta(days=60):
        raise ValueError('Choose a past aware GONG range up to 60 days')
    source = (policy.sources.historical if historical else policy.sources.live)[0]
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    async with get_session_factory()() as session:
        existing = set((await session.scalars(select(GONGSnapshot.observed_at).where(
            GONGSnapshot.observed_at >= start, GONGSnapshot.observed_at <= end))).all())
    latest = max(existing) if existing else None
    report['retained'] = len(existing)
    try:
        candidates = await asyncio.to_thread(provider.candidates, start, end, historical=historical)
        # One magnetogram per observed hour; keep already received hours immutable.
        hours = {t.replace(minute=0, second=0, microsecond=0) for t in existing}
        for observed, url in candidates:
            hour = observed.replace(minute=0, second=0, microsecond=0)
            if hour in hours:
                continue
            try:
                values = await asyncio.to_thread(snapshot_values, observed, url, source)
                async with get_session_factory()() as session:
                    statement = insert(GONGSnapshot).values(**values).on_conflict_do_nothing(
                        index_elements=['slot_at']).returning(GONGSnapshot.observed_at)
                    inserted = (await session.execute(statement)).scalar_one_or_none()
                    await session.commit()
                report['received' if inserted else 'retained'] += 1
                hours.add(hour)
                if inserted:
                    latest = max(latest, observed) if latest else observed
            except Exception as exc:
                report['failed'] += 1
                if len(report['source_errors']) < 20:
                    report['source_errors'].append({'product': source, 'error': str(exc)})
        expected = max(1, int((end-start).total_seconds() // 3600))
        report['missing'] = max(0, expected-len(hours))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': source, 'error': str(exc)})
    report['latest_observed_at'] = latest.isoformat() if latest else None
    failed = report['failed'] > 0 if historical else latest is None or latest < now-timedelta(hours=3)
    return {'status': 'partial' if failed or report['missing'] or report['failed'] else 'complete',
            'failed_metrics': ['gong'] if failed else [], 'files': {'gong': report}}
