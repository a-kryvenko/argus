"""Read-only model input freshness alongside native collector diagnostics."""
from datetime import UTC, datetime
from sqlalchemy import select
from argus_clio.db.models import Measurement, MeasurementReceipt, ScheduledJob
from clio.observations import OBSERVATION_METRICS
from argus_clio.services.collection_status import source_status

async def monitoring_status(session):
    now = datetime.now(UTC)
    result = await source_status(session, now)
    receipts = {r.metric: r for r in (await session.scalars(select(MeasurementReceipt))).all()}
    measurements = []
    for metric in OBSERVATION_METRICS:
        latest = await session.scalar(select(Measurement.observed_at).where(Measurement.metric == metric)
                                      .order_by(Measurement.observed_at.desc()).limit(1))
        receipt = receipts.get(metric)
        age = (now - latest).total_seconds() if latest else None
        threshold = 72 * 3600 if metric in {'f10_7', 's10', 'm10', 'y10'} else 6 * 3600
        measurements.append({'metric': metric, 'latest_observation_at': latest,
            'received_at': receipt.received_at if receipt and receipt.latest_observation_at == latest else None,
            'age_seconds': age, 'stale_after_seconds': threshold,
            'status': 'unavailable' if latest is None else 'future' if age < -300 else 'delayed' if age > threshold else 'fresh'})
    job = await session.get(ScheduledJob, 'refresh')
    result['measurements'] = measurements
    result['last_refresh_completed_at'] = job.completed_at if job else None
    if any(m['status'] != 'fresh' for m in measurements):
        result['status'] = 'degraded'
    return result
